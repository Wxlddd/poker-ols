import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pokerkit import HandHistory
import warnings

# Suppress pokerkit and matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def get_starting_hand(card_str):
    rank_order = 'AKQJT98765432'
    if len(card_str) != 4 or '?' in card_str:
        return None
    c1, c2 = card_str[0:2], card_str[2:4]
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    
    try:
        idx1 = rank_order.index(r1)
        idx2 = rank_order.index(r2)
    except ValueError:
        return None
        
    if idx1 == idx2:
        return f"{r1}{r2}"
    elif idx1 < idx2:
        suffix = 's' if s1 == s2 else 'o'
        return f"{r1}{r2}{suffix}"
    else:
        suffix = 's' if s1 == s2 else 'o'
        return f"{r2}{r1}{suffix}"

def analyze_hands():
    print("=" * 70)
    print("STARTING HAND PROFITABILITY & STRATEGY STUDY")
    print("=" * 70)
    
    files = glob.glob(os.path.join("scratch", "abs_NLH_handhq_*.phhs"))
    # Sort files numerically
    files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    if not files:
        print("Error: No hand history files found in scratch/.")
        return
        
    print(f"Found {len(files)} hand history files to process.")
    
    # Track stats per hand notation
    # 'AA' -> {'count': int, 'profit_bb': float, 'vpip': int, 'pfr': int}
    hand_stats = {}
    
    # Initialize all 169 Hold'em hands to ensure they are present in our dataset
    rank_order = 'AKQJT98765432'
    for i in range(13):
        for j in range(13):
            r1, r2 = rank_order[i], rank_order[j]
            if i == j:
                name = f"{r1}{r2}"
            elif i < j:
                name = f"{r1}{r2}s"
            else:
                name = f"{r2}{r1}o"
            hand_stats[name] = {'count': 0, 'profit_bb': 0.0, 'vpip': 0, 'pfr': 0}
            
    total_hands_parsed = 0
    known_hole_cards_count = 0
    
    for file_idx, file_path in enumerate(files, 1):
        print(f"[{file_idx}/{len(files)}] Parsing {os.path.basename(file_path)}...", end="", flush=True)
        file_hands = 0
        
        try:
            with open(file_path, "rb") as f:
                for hh in HandHistory.load_all(f):
                    # Blinds & BB value
                    blinds = hh.blinds_or_straddles
                    bb_val = max(blinds)
                    if bb_val <= 0:
                        continue
                    bb_idx = blinds.index(bb_val)
                    
                    # Players in the hand
                    players = hh.players
                    num_players = len(players)
                    
                    # Preflop statistics tracking for this hand
                    preflop_raises = 1 # BB is the first bet
                    vpip_in_hand = [False] * num_players
                    pfr_in_hand = [False] * num_players
                    
                    flop_dealt = False
                    for action in hh.actions:
                        if action.startswith("d db "):
                            flop_dealt = True
                            break
                            
                    in_postflop = False
                    for action in hh.actions:
                        if action.startswith("d db "):
                            in_postflop = True
                            continue
                        if action.startswith("d "):
                            continue
                        parts = action.split()
                        if len(parts) < 2:
                            continue
                        p_idx = int(parts[0][1:]) - 1
                        if p_idx < 0 or p_idx >= num_players:
                            continue
                        act_type = parts[1]
                        
                        if not in_postflop:
                            if act_type == "cbr":
                                preflop_raises += 1
                                vpip_in_hand[p_idx] = True
                                pfr_in_hand[p_idx] = True
                            elif act_type == "cc":
                                if p_idx == bb_idx:
                                    if preflop_raises > 1:
                                        vpip_in_hand[p_idx] = True
                                else:
                                    vpip_in_hand[p_idx] = True
                                    
                    # Payoffs simulation
                    state = None
                    for s, act in hh.state_actions:
                        state = s
                    payoffs = state.payoffs
                    
                    # Find hole cards in actions (dealt or shown at showdown)
                    hole_cards = [None] * num_players
                    for action in hh.actions:
                        if action.startswith("d dh p"):
                            parts = action.split()
                            p_idx = int(parts[2][1:]) - 1
                            cards = parts[3]
                            if cards != "????" and len(cards) == 4:
                                hole_cards[p_idx] = cards
                        elif " sm " in action:
                            # e.g., p1 sm Ac2d
                            parts = action.split()
                            p_idx = int(parts[0][1:]) - 1
                            cards = parts[2]
                            if cards != "????" and len(cards) == 4:
                                hole_cards[p_idx] = cards
                                
                    # Record hand-level stats for each known starting hand
                    for idx, cards in enumerate(hole_cards):
                        if cards:
                            hand_notation = get_starting_hand(cards)
                            if hand_notation:
                                payoff_bb = float(payoffs[idx]) / float(bb_val)
                                
                                # Accumulate
                                hand_stats[hand_notation]['count'] += 1
                                hand_stats[hand_notation]['profit_bb'] += payoff_bb
                                if vpip_in_hand[idx]:
                                    hand_stats[hand_notation]['vpip'] += 1
                                if pfr_in_hand[idx]:
                                    hand_stats[hand_notation]['pfr'] += 1
                                known_hole_cards_count += 1
                                
                    file_hands += 1
                    total_hands_parsed += 1
                    
            print(f" Done ({file_hands} hands parsed, {known_hole_cards_count} known hole cards).")
        except Exception as e:
            print(f" Failed: {e}")
            
    print("\n" + "=" * 70)
    print("PARSING COMPLETE!")
    print(f"Total hands parsed: {total_hands_parsed}")
    print(f"Total known starting hand observations: {known_hole_cards_count}")
    print("=" * 70)
    
    # Convert stats to a DataFrame
    records = []
    for hand, stats in hand_stats.items():
        count = stats['count']
        profit_bb = stats['profit_bb']
        vpip = stats['vpip']
        pfr = stats['pfr']
        
        avg_profit = profit_bb / count if count > 0 else 0.0
        vpip_pct = (vpip / count) * 100 if count > 0 else 0.0
        pfr_pct = (pfr / count) * 100 if count > 0 else 0.0
        
        records.append({
            'Hand': hand,
            'Count': count,
            'TotalProfitBB': profit_bb,
            'AvgProfitBB': avg_profit,
            'VPIP%': vpip_pct,
            'PFR%': pfr_pct
        })
        
    df = pd.DataFrame(records)
    
    # Save the raw data
    df.to_csv("starting_hand_stats.csv", index=False)
    print("Saved hand statistics to 'starting_hand_stats.csv'.")
    
    # Print Top 15 most profitable hands
    print("\n--- Top 15 Most Profitable Observed Starting Hands ---")
    top_profitable = df[df['Count'] >= 10].sort_values(by='AvgProfitBB', ascending=False).head(15)
    print(top_profitable[['Hand', 'Count', 'AvgProfitBB', 'VPIP%', 'PFR%']].to_string(index=False, formatters={
        'AvgProfitBB': '{:+.4f} BB'.format, 'VPIP%': '{:.1f}%'.format, 'PFR%': '{:.1f}%'.format
    }))
    
    # Print Top 15 most losing hands
    print("\n--- Top 15 Most Losing Observed Starting Hands ---")
    top_losing = df[df['Count'] >= 10].sort_values(by='AvgProfitBB', ascending=True).head(15)
    print(top_losing[['Hand', 'Count', 'AvgProfitBB', 'VPIP%', 'PFR%']].to_string(index=False, formatters={
        'AvgProfitBB': '{:+.4f} BB'.format, 'VPIP%': '{:.1f}%'.format, 'PFR%': '{:.1f}%'.format
    }))
    
    # ----------------------------------------------------
    # Generate the 13x13 Heatmap Matrix
    # ----------------------------------------------------
    profit_matrix = np.zeros((13, 13))
    count_matrix = np.zeros((13, 13))
    label_matrix = np.empty((13, 13), dtype=object)
    
    for i in range(13):
        for j in range(13):
            r1, r2 = rank_order[i], rank_order[j]
            if i == j:
                hand_name = f"{r1}{r2}"
            elif i < j:
                hand_name = f"{r1}{r2}s"
            else:
                hand_name = f"{r2}{r1}o"
                
            stats = hand_stats[hand_name]
            avg_profit = stats['profit_bb'] / stats['count'] if stats['count'] > 0 else 0.0
            
            profit_matrix[i, j] = avg_profit
            count_matrix[i, j] = stats['count']
            
            # Label contains hand name and profit
            label_matrix[i, j] = f"{hand_name}\n{avg_profit:+.2f}"
            
    # Professional Heatmap Rendering
    plt.figure(figsize=(15, 14))
    
    # Custom Diverging Color Map: Red (losses) -> White (neutral) -> Green (wins)
    # Using subtle, elegant colors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "premium_poker_profits", 
        ["#e74c3c", "#fcfcfc", "#2ecc71"],
        N=256
    )
    
    # Find symmetric range around zero for balanced color mapping
    max_val = max(abs(np.min(profit_matrix)), abs(np.max(profit_matrix)))
    # Cap at a reasonable value so outliers don't wash out the plot (e.g., limit scale to +/- 1.5 BB per hand)
    v_limit = min(max_val, 1.5)
    
    sns.set_theme(style="white")
    
    # Plot heatmap
    ax = sns.heatmap(
        profit_matrix,
        annot=label_matrix,
        fmt="",
        cmap=cmap,
        center=0.0,
        vmin=-v_limit,
        vmax=v_limit,
        linewidths=0.5,
        linecolor="#bdc3c7",
        xticklabels=list(rank_order),
        yticklabels=list(rank_order),
        cbar_kws={'label': 'Average Profit per Hand (Big Blinds)', 'shrink': 0.8},
        annot_kws={"size": 9, "weight": "bold"}
    )
    
    # Adjust aesthetics
    plt.title("Empirical Hold'em Starting Hand Profitability Grid\n(Diagonal: Pairs | Upper-Right: Suited | Lower-Left: Offsuit)\nAnnotated with: Hand & Average Profit in Big Blinds", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Second Card / Suited Card Rank", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("First Card / Offsuit Card Rank", fontsize=12, fontweight='bold', labelpad=10)
    
    # Style tick marks
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Highlight the diagonal (Pocket Pairs) with a subtle box
    for i in range(13):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='#2c3e50', lw=2))
        
    plt.tight_layout()
    heatmap_path = "starting_hand_heatmap.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nStunning starting hand heatmap saved to '{heatmap_path}'.")

if __name__ == '__main__':
    analyze_hands()
