import os
import numpy as np
import pandas as pd

def generate_realistic_poker_data(n_players=10000, seed=42):
    """
    Generates a highly realistic poker statistics dataset.
    Poker player stats follow well-known joint distributions:
    - Hands: Power-law or log-normal (many recreational players with few hands, few regulars with many hands).
    - VPIP: Log-normal or Beta (most players between 15% and 35%, some extreme maniacs/fish at 50%+).
    - PFR: Strictly bounded by VPIP (PFR <= VPIP). Good players have PFR close to VPIP. Passive players have VPIP >> PFR.
    - 3Bet: Positively correlated with PFR, usually 3Bet <= PFR. Typical range 2% to 15%.
    - Win Rate (bb/100): Function of VPIP, PFR, 3Bet, with variance inversely proportional to Hands (formulaic poker variance).
    """
    np.random.seed(seed)
    
    # 1. Generate Hands (log-normal distribution)
    # Most players have few hands, a few regulars have huge volume.
    # We want a range from 100 hands to 1,000,000 hands.
    log_hands = np.random.lognormal(mean=7.5, sigma=1.5, size=n_players)
    hands = np.clip(log_hands, 100, 1000000).astype(int)
    
    # 2. Generate VPIP (Voluntary Put Money in Pot %)
    # Normal/Beta-like distribution centered around 25% VPIP.
    vpip = np.random.beta(a=4, b=12, size=n_players) * 100  # Shift and scale
    # Ensure realistic range [5%, 85%]
    vpip = np.clip(vpip, 5.0, 85.0)
    
    # 3. Generate PFR (Preflop Raise %)
    # PFR is strictly <= VPIP.
    # The ratio PFR/VPIP represents aggressiveness. Good players have ratio ~0.7-0.9. Passive players have ratio ~0.2-0.5.
    pfr_ratio = np.random.beta(a=6, b=3, size=n_players)  # Leans towards higher ratio, but with a tail
    # Adjust ratio for loose players (maniacs or fish have lower ratio)
    pfr_ratio = pfr_ratio * (1.0 - (vpip / 120.0))  # higher VPIP players tend to be more passive (higher gap)
    pfr_ratio = np.clip(pfr_ratio, 0.1, 0.95)
    
    pfr = vpip * pfr_ratio
    pfr = np.clip(pfr, 1.0, vpip - 1.0)  # PFR must be less than VPIP
    
    # 4. Generate 3Bet % (Three-bet preflop %)
    # Heavily correlated with PFR. Usually 3Bet is around 30% to 50% of PFR, capped realistically.
    three_bet_ratio = np.random.normal(loc=0.35, scale=0.08, size=n_players)
    three_bet_ratio = np.clip(three_bet_ratio, 0.1, 0.6)
    three_bet = pfr * three_bet_ratio
    three_bet = np.clip(three_bet, 0.5, 20.0)  # 3Bet between 0.5% and 20%
    
    # 5. Generate Win Rate (bb/100)
    # The true winrate (skill) is a non-linear or linear function of VPIP, PFR, and 3Bet.
    # In poker theory, optimal stats are around VPIP=22, PFR=18, 3Bet=8 (TAG style).
    # Extreme tightness (VPIP < 15) is slightly losing/break-even due to blinds.
    # Extreme looseness (VPIP > 35) is heavily losing, especially if passive (low PFR).
    # High gap (VPIP - PFR) is heavily penalized (passive play).
    # Let's model a realistic true winrate function:
    
    # Base winrate centered around optimal stats
    vpip_dist_from_opt = np.abs(vpip - 22.0)
    pfr_dist_from_opt = np.abs(pfr - 17.0)
    gap = vpip - pfr
    
    # True skill win rate (long-term bb/100)
    true_winrate = (
        2.5  # Base win rate for a perfect player
        - 0.15 * vpip_dist_from_opt  # Penalize deviating from optimal VPIP
        - 0.20 * pfr_dist_from_opt   # Penalize deviating from optimal PFR
        - 0.40 * (gap - 4.0)          # Penalize large VPIP-PFR gaps (passivity)
        + 0.30 * (three_bet - 6.0)    # Slight reward for active 3-betting, capped
    )
    
    # Cap the extreme negative/positive true winrates to realistic values
    true_winrate = np.clip(true_winrate, -40.0, 10.0)
    
    # Short-term variance (standard deviation of bb/100 in a single hand is about 80 bb/100)
    # Standard deviation over N hands is: 80 * sqrt(100 / N) = 800 / sqrt(N)
    std_err = 80.0 / np.sqrt(hands / 100.0)
    
    # Observed winrate = True winrate + Noise (sample-size dependent variance)
    observed_winrate = true_winrate + np.random.normal(loc=0.0, scale=std_err, size=n_players)
    observed_winrate = np.round(observed_winrate, 2)
    
    # Round percentages to 1 decimal place
    vpip = np.round(vpip, 1)
    pfr = np.round(pfr, 1)
    three_bet = np.round(three_bet, 1)
    
    df = pd.DataFrame({
        'Player': [f'Player_{i}' for i in range(n_players)],
        'Hands': hands,
        'VPIP': vpip,
        'PFR': pfr,
        '3Bet': three_bet,
        'WinRate': observed_winrate
    })
    
    return df

if __name__ == '__main__':
    print("Generating a highly realistic poker player statistics dataset...")
    df = generate_realistic_poker_data()
    output_file = 'poker_stats.csv'
    df.to_csv(output_file, index=False)
    print(f"Dataset successfully saved as '{output_file}' with {len(df)} players.")
    print("\nDataset summary statistics:")
    print(df.describe())
