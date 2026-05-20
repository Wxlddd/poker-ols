import urllib.request
import os
import pandas as pd
import numpy as np
from pokerkit import HandHistory
import warnings

# Suppress UserWarnings from pokerkit
warnings.filterwarnings("ignore", category=UserWarning)

def download_file(file_index):
    file_name = f"abs_NLH_handhq_{file_index}.phhs"
    local_path = os.path.join("scratch", file_name)
    
    if os.path.exists(local_path):
        return local_path
        
    url = f"https://raw.githubusercontent.com/uoftcprg/phh-dataset/main/data/handhq/ABS-2009-07-01_2009-07-23_100NLH_OBFU/1/abs%20NLH%20handhq_{file_index}-OBFUSCATED.phhs"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()
            with open(local_path, "wb") as f:
                f.write(content)
        return local_path
    except Exception as e:
        print(f"Failed to download file {file_index}: {e}")
        return None

def build_dataset(num_files=40):
    print("=" * 60)
    print(f"BUILDING POKER PLAYER DATASET FROM REAL HAND HISTORIES ({num_files} files)")
    print("=" * 60)
    
    os.makedirs("scratch", exist_ok=True)
    
    player_data = {} # player_name -> {hands, vpip, pfr, threebet, profit_bb}
    
    hands_processed = 0
    
    for i in range(1, num_files + 1):
        print(f"Processing file {i}/{num_files}...", end="", flush=True)
        local_file = download_file(i)
        if not local_file:
            print(" skipped (download failed).")
            continue
            
        file_hands = 0
        try:
            with open(local_file, "rb") as f:
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
                    threebet_in_hand = [False] * num_players
                    
                    # Parse actions
                    for action in hh.actions:
                        if not action.startswith("p") and not action.startswith("d dh"):
                            break
                        if action.startswith("d "):
                            continue
                        parts = action.split()
                        if len(parts) < 2:
                            continue
                        p_idx = int(parts[0][1:]) - 1
                        if p_idx < 0 or p_idx >= num_players:
                            continue
                        act_type = parts[1]
                        
                        if act_type == "cbr":
                            preflop_raises += 1
                            vpip_in_hand[p_idx] = True
                            pfr_in_hand[p_idx] = True
                            if preflop_raises == 3:
                                threebet_in_hand[p_idx] = True
                        elif act_type == "cc":
                            if p_idx == bb_idx:
                                if preflop_raises > 1:
                                    vpip_in_hand[p_idx] = True
                            else:
                                vpip_in_hand[p_idx] = True
                    
                    # Payouts simulation
                    state = None
                    for s, act in hh.state_actions:
                        state = s
                    payoffs = state.payoffs
                    
                    # Update global player statistics
                    for idx, name in enumerate(players):
                        if name not in player_data:
                            player_data[name] = {
                                "hands": 0,
                                "vpip": 0,
                                "pfr": 0,
                                "threebet": 0,
                                "profit_bb": 0.0
                            }
                        
                        # Accumulate
                        p_stats = player_data[name]
                        p_stats["hands"] += 1
                        if vpip_in_hand[idx]:
                            p_stats["vpip"] += 1
                        if pfr_in_hand[idx]:
                            p_stats["pfr"] += 1
                        if threebet_in_hand[idx]:
                            p_stats["threebet"] += 1
                        
                        # Profit in Big Blinds
                        payoff = payoffs[idx]
                        p_stats["profit_bb"] += float(payoff) / float(bb_val)
                    
                    file_hands += 1
                    hands_processed += 1
            
            print(f" Done ({file_hands} hands parsed).")
        except Exception as e:
            print(f" Failed to parse: {e}")
            
    print("\nProcessing complete!")
    print(f"Total hands processed: {hands_processed}")
    print(f"Total unique players found: {len(player_data)}")
    
    # Convert to DataFrame
    rows = []
    for name, stats in player_data.items():
        hands = stats["hands"]
        if hands == 0:
            continue
        vpip_pct = (stats["vpip"] / hands) * 100
        pfr_pct = (stats["pfr"] / hands) * 100
        threebet_pct = (stats["threebet"] / hands) * 100
        winrate_bb_100 = (stats["profit_bb"] * 100) / hands
        
        rows.append({
            "Player": name,
            "Hands": hands,
            "VPIP": vpip_pct,
            "PFR": pfr_pct,
            "3Bet": threebet_pct,
            "WinRate": winrate_bb_100
        })
        
    df = pd.DataFrame(rows)
    
    # Save unfiltered dataset
    df.to_csv("poker_stats.csv", index=False)
    print("\nDataset successfully saved to 'poker_stats.csv'!")
    
    # Display statistics of sample sizes
    print("\n--- Player Distribution by Hand Count ---")
    thresholds = [1, 10, 50, 100, 500, 1000, 2000, 5000]
    for t in thresholds:
        count = len(df[df['Hands'] > t])
        print(f"Players with Hands > {t:4d}: {count}")
        
    return df

if __name__ == '__main__':
    build_dataset(40)
