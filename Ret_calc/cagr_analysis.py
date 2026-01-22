import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from Home_buying_calc import calculate_deal, get_default_parameters

# Define the deals
deal1 = {
    'deal_name': 'Irent_urent',
    'sqft': 2300,
    'quote_price': 500000.00,
    'fair_market_value': 500000.00,
    'rent_per_sqft': 1.31,
    'down_percent': 0.20,
    'advance_payment': 5000.00,
    'one_time_other': 20000.00,
    'closing_credits': 0.00,
    'loan_duration_years': 30,
    'base_interest_rate': 0.04,
    'points_purchased': 0,
    'cost_per_point_percent': 1.0,
    'rate_reduction_per_point': 0.0025,
    'prop_tax_rate': 0.0151,
    'hoa_monthly': 119.00,
    'home_ins_monthly': 60.00,
    'pmi_monthly': 0.00,
    'prop_management_percent': 0.0,
    'safety_deposit_months': 2,
    'holding_years': 5,
    'appreciation_yoy': 0.03,
    'realtor_commission_factor': 0.94,
    'tax_bracket': 0.24,
    'state': 'PA',
    'long_term_cap_gains_rate': 0.15,
    'depreciation_years': 27.5,
    'building_value_ratio': 0.80,
    'recapture_tax_rate': 0.25,
    'sp500_annual_return': 0.10
}

deal2 = {
    'deal_name': 'DR_horton_unit',
    'sqft': 2035,
    'quote_price': 370000.00,
    'fair_market_value': 410000.00,
    'rent_per_sqft': 1.0,
    'down_percent': 0.25,
    'advance_payment': 5000.00,
    'one_time_other': 7000.00,
    'closing_credits': 0.00,
    'loan_duration_years': 30,
    'base_interest_rate': 0.065,
    'points_purchased': 0,
    'cost_per_point_percent': 1.0,
    'rate_reduction_per_point': 0.0025,
    'prop_tax_rate': 0.009,
    'hoa_monthly': 29.167,
    'home_ins_monthly': 125.00,
    'pmi_monthly': 0.00,
    'prop_management_percent': 0.007,
    'safety_deposit_months': 2,
    'holding_years': 5,
    'appreciation_yoy': 0.03,
    'realtor_commission_factor': 0.94,
    'tax_bracket': 0.24,
    'state': 'AR',
    'long_term_cap_gains_rate': 0.15,
    'depreciation_years': 27.5,
    'building_value_ratio': 0.80,
    'recapture_tax_rate': 0.25,
    'sp500_annual_return': 0.10
}

my_deals = [deal1, deal2]

# Analysis parameters
holding_years_range = np.arange(1, 11)  # 1-10 years
appreciation_rates = np.array([-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])  # -2% to +5%

# Create figure with subplots: 2 rows (one per deal), 2 columns (years vs appreciation)
fig, axes = plt.subplots(len(my_deals), 2, figsize=(16, 10))
fig.suptitle('CAGR Analysis: Impact of Holding Period and Appreciation Rate', fontsize=16, fontweight='bold')

print("Generating CAGR line plots...")

for deal_idx, deal in enumerate(my_deals):
    # =========================================================================
    # LEFT COLUMN: CAGR vs Holding Years (with different appreciation rates)
    # =========================================================================
    ax_left = axes[deal_idx, 0]
    
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(appreciation_rates)))
    
    for app_idx, appreciation in enumerate(appreciation_rates):
        cagrs = []
        
        for years in holding_years_range:
            deal_copy = deal.copy()
            deal_copy['holding_years'] = years
            deal_copy['appreciation_yoy'] = appreciation
            
            result = calculate_deal(deal_copy)
            cagrs.append(result['annualized_return'])
        
        label = f"{appreciation*100:+.0f}%" 
        ax_left.plot(holding_years_range, cagrs, marker='o', label=label, 
                    linewidth=2, color=colors[app_idx])
    
    ax_left.set_xlabel('Holding Period (Years)', fontsize=11, fontweight='bold')
    ax_left.set_ylabel('CAGR (%)', fontsize=11, fontweight='bold')
    ax_left.set_title(f'{deal["deal_name"]} - CAGR vs Holding Years', fontsize=12, fontweight='bold')
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(title='Appreciation Rate', fontsize=9, title_fontsize=10)
    ax_left.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # =========================================================================
    # RIGHT COLUMN: CAGR vs Appreciation Rate (with different holding years)
    # =========================================================================
    ax_right = axes[deal_idx, 1]
    
    years_to_show = [1, 3, 5, 7, 10]
    colors_years = plt.cm.viridis(np.linspace(0, 1, len(years_to_show)))
    
    for year_idx, years in enumerate(years_to_show):
        cagrs = []
        
        for appreciation in appreciation_rates:
            deal_copy = deal.copy()
            deal_copy['holding_years'] = years
            deal_copy['appreciation_yoy'] = appreciation
            
            result = calculate_deal(deal_copy)
            cagrs.append(result['annualized_return'])
        
        ax_right.plot(appreciation_rates * 100, cagrs, marker='s', label=f'{years} yr(s)', 
                     linewidth=2, color=colors_years[year_idx])
    
    ax_right.set_xlabel('Appreciation Rate (%)', fontsize=11, fontweight='bold')
    ax_right.set_ylabel('CAGR (%)', fontsize=11, fontweight='bold')
    ax_right.set_title(f'{deal["deal_name"]} - CAGR vs Appreciation Rate', fontsize=12, fontweight='bold')
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(title='Holding Period', fontsize=9, title_fontsize=10)
    ax_right.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('cagr_analysis.png', dpi=300, bbox_inches='tight')
print("✓ CAGR graph saved as 'cagr_analysis.png'")
plt.close()

# =========================================================================
# ROI LINE PLOTS: Similar structure to CAGR
# =========================================================================
fig, axes = plt.subplots(len(my_deals), 2, figsize=(16, 10))
fig.suptitle('ROI Analysis: Impact of Holding Period and Appreciation Rate', fontsize=16, fontweight='bold')

print("Generating ROI line plots...")

for deal_idx, deal in enumerate(my_deals):
    # =========================================================================
    # LEFT COLUMN: ROI vs Holding Years (with different appreciation rates)
    # =========================================================================
    ax_left = axes[deal_idx, 0]
    
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(appreciation_rates)))
    
    for app_idx, appreciation in enumerate(appreciation_rates):
        rois = []
        
        for years in holding_years_range:
            deal_copy = deal.copy()
            deal_copy['holding_years'] = years
            deal_copy['appreciation_yoy'] = appreciation
            
            result = calculate_deal(deal_copy)
            rois.append(result['roi'])
        
        label = f"{appreciation*100:+.0f}%" 
        ax_left.plot(holding_years_range, rois, marker='o', label=label, 
                    linewidth=2, color=colors[app_idx])
    
    ax_left.set_xlabel('Holding Period (Years)', fontsize=11, fontweight='bold')
    ax_left.set_ylabel('ROI (%)', fontsize=11, fontweight='bold')
    ax_left.set_title(f'{deal["deal_name"]} - ROI vs Holding Years', fontsize=12, fontweight='bold')
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(title='Appreciation Rate', fontsize=9, title_fontsize=10)
    ax_left.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # =========================================================================
    # RIGHT COLUMN: ROI vs Appreciation Rate (with different holding years)
    # =========================================================================
    ax_right = axes[deal_idx, 1]
    
    years_to_show = [1, 3, 5, 7, 10]
    colors_years = plt.cm.viridis(np.linspace(0, 1, len(years_to_show)))
    
    for year_idx, years in enumerate(years_to_show):
        rois = []
        
        for appreciation in appreciation_rates:
            deal_copy = deal.copy()
            deal_copy['holding_years'] = years
            deal_copy['appreciation_yoy'] = appreciation
            
            result = calculate_deal(deal_copy)
            rois.append(result['roi'])
        
        ax_right.plot(appreciation_rates * 100, rois, marker='s', label=f'{years} yr(s)', 
                     linewidth=2, color=colors_years[year_idx])
    
    ax_right.set_xlabel('Appreciation Rate (%)', fontsize=11, fontweight='bold')
    ax_right.set_ylabel('ROI (%)', fontsize=11, fontweight='bold')
    ax_right.set_title(f'{deal["deal_name"]} - ROI vs Appreciation Rate', fontsize=12, fontweight='bold')
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(title='Holding Period', fontsize=9, title_fontsize=10)
    ax_right.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('roi_analysis.png', dpi=300, bbox_inches='tight')
print("✓ ROI graph saved as 'roi_analysis.png'")
plt.close()

# =========================================================================
# Create a detailed heatmap for CAGR for each deal
# =========================================================================
fig, axes = plt.subplots(1, len(my_deals), figsize=(16, 6))
fig.suptitle('CAGR Heatmap: Holding Years vs Appreciation Rate', fontsize=16, fontweight='bold')

print("Generating CAGR heatmap...")

holding_years_range_heat = np.arange(1, 11)  # 1-10 years

for deal_idx, deal in enumerate(my_deals):
    cagr_matrix = np.zeros((len(holding_years_range_heat), len(appreciation_rates)))
    
    for year_idx, years in enumerate(holding_years_range_heat):
        for app_idx, appreciation in enumerate(appreciation_rates):
            deal_copy = deal.copy()
            deal_copy['holding_years'] = years
            deal_copy['appreciation_yoy'] = appreciation
            
            result = calculate_deal(deal_copy)
            cagr_matrix[year_idx, app_idx] = result['annualized_return']
    
    ax = axes[deal_idx]
    norm = mcolors.CenteredNorm(vcenter=0)
    im = ax.imshow(cagr_matrix, cmap='RdYlGn', aspect='auto', origin='lower', norm=norm)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(appreciation_rates)))
    ax.set_yticks(np.arange(len(holding_years_range_heat)))
    ax.set_xticklabels([f'{x*100:+.0f}%' for x in appreciation_rates], rotation=45)
    ax.set_yticklabels([f'{int(y)} yr' for y in holding_years_range_heat])
    
    ax.set_xlabel('Appreciation Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Holding Years', fontsize=11, fontweight='bold')
    ax.set_title(f'{deal["deal_name"]}', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for year_idx in range(len(holding_years_range_heat)):
        for app_idx in range(len(appreciation_rates)):
            text = ax.text(app_idx, year_idx, f'{cagr_matrix[year_idx, app_idx]:.1f}%',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CAGR (%)', fontweight='bold')

plt.tight_layout()
plt.savefig('cagr_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ CAGR heatmap saved as 'cagr_heatmap.png'")
plt.close()

# =========================================================================
# Create a detailed heatmap for ROI for each deal
# =========================================================================
fig, axes = plt.subplots(1, len(my_deals), figsize=(16, 6))
fig.suptitle('ROI Heatmap: Holding Years vs Appreciation Rate', fontsize=16, fontweight='bold')

print("Generating ROI heatmap...")

holding_years_range_heat = np.arange(1, 11)  # 1-10 years

for deal_idx, deal in enumerate(my_deals):
    roi_matrix = np.zeros((len(holding_years_range_heat), len(appreciation_rates)))
    
    for year_idx, years in enumerate(holding_years_range_heat):
        for app_idx, appreciation in enumerate(appreciation_rates):
            deal_copy = deal.copy()
            deal_copy['holding_years'] = years
            deal_copy['appreciation_yoy'] = appreciation
            
            result = calculate_deal(deal_copy)
            roi_matrix[year_idx, app_idx] = result['roi']
    
    ax = axes[deal_idx]
    norm = mcolors.CenteredNorm(vcenter=0)
    im = ax.imshow(roi_matrix, cmap='RdYlGn', aspect='auto', origin='lower', norm=norm)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(appreciation_rates)))
    ax.set_yticks(np.arange(len(holding_years_range_heat)))
    ax.set_xticklabels([f'{x*100:+.0f}%' for x in appreciation_rates], rotation=45)
    ax.set_yticklabels([f'{int(y)} yr' for y in holding_years_range_heat])
    
    ax.set_xlabel('Appreciation Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Holding Years', fontsize=11, fontweight='bold')
    ax.set_title(f'{deal["deal_name"]}', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for year_idx in range(len(holding_years_range_heat)):
        for app_idx in range(len(appreciation_rates)):
            text = ax.text(app_idx, year_idx, f'{roi_matrix[year_idx, app_idx]:.1f}%',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('ROI (%)', fontweight='bold')

plt.tight_layout()
plt.savefig('roi_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ ROI heatmap saved as 'roi_heatmap.png'")
plt.close()

print("\n✓ Analysis complete! Four graphs have been generated:")
print("  1. cagr_analysis.png - CAGR line plots")
print("  2. roi_analysis.png - ROI line plots")
print("  3. cagr_heatmap.png - CAGR heatmaps")
print("  4. roi_heatmap.png - ROI heatmaps")
