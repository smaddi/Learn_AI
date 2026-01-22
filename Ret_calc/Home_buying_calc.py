import math

def get_default_parameters():
    """Returns the default configuration for a home deal."""
    return {
        'deal_name': 'Default Deal',
        
        # Property Details
        'sqft': 2300,
        'quote_price': 500000.00,       # C4: Purchase Price
        'fair_market_value': 500000.00, # Purchase Price (Default)
        'rent_per_sqft': 1.31,
        
        # Loan & Investment Details
        'down_percent': 0.20,           # C5: 20%
        'advance_payment': 5000.00,     # C6: Pre-paid/Advance
        'one_time_other': 20000.00,     # C12: Closing costs/Repairs
        'closing_credits': 0.00,        # Credit received at closing
        
        # Mortgage Buy Down
        'loan_duration_years': 30,      # Standard 30 years
        'base_interest_rate': 0.04,     # Current Market Rate
        'points_purchased': 0,          # Number of points to buy
        'cost_per_point_percent': 1.0,  # Cost per point as % of loan amount
        'rate_reduction_per_point': 0.0025, # 0.25% reduction per point
        
        # Expenses & Reserves
        'prop_tax_rate': 0.0151,        # J9: 1.51%
        'hoa_monthly': 119.00,          # J11
        'home_ins_monthly': 60.00,      # J12
        'pmi_monthly': 0.00,            # J10
        'prop_management_percent': 0.0, # 8% of rent
        'safety_deposit_months': 2,     # Reserve fund in months of rent
        
        # Growth & Taxes
        'holding_years': 5,             # J3: Years to own
        'appreciation_yoy': 0.03,       # J16: 3%
        'rent_appreciation_yoy': 0.03,  # YoY Rent growth
        'expense_appreciation_yoy': 0.02, # YoY Expense growth
        'realtor_commission_factor': 0.94, # J18: 100% - 6% commission = 94% net
        'tax_bracket': 0.24,            # J25: 24% (Federal Ordinary Income)
        
        # State Selection
        'state': 'PA',                  # Options: 'CA', 'TX', 'FL', 'NY', 'WA', 'IL', 'MA', 'PA'
        
        # Investment Tax Details
        'long_term_cap_gains_rate': 0.15, # Federal Long Term
        'depreciation_years': 27.5,       # Residential rental property standard
        'building_value_ratio': 0.80,     # 80% building value (depreciable)
        'recapture_tax_rate': 0.25,       # Depreciation recapture tax rate (Federal)
        
        # S&P 500 Comparison Details
        'sp500_annual_return': 0.10,      # 10% average annual return
        
        # Cashflow Strategy
        'positive_cashflow_strategy': 'reinvest' # Options: 'reinvest', 'pay_down_loan'
    }

# Mortgage types are now defined directly within each deal using 'loan_scenarios'.

def calculate_deal(params):
    """
    Calculates all metrics for a given deal configuration dictionary.
    Returns a dictionary containing all input parameters and calculated results.
    """
    
    # Merge input params with defaults to ensure all keys exist
    p = get_default_parameters()
    p.update(params)
    
    # Extract values for cleaner math code below
    sqft = p['sqft']
    quote_price = p['quote_price']
    rent_per_sqft = p['rent_per_sqft']
    
    down_percent = p['down_percent']
    advance_payment = p['advance_payment']
    one_time_other = p['one_time_other']
    closing_credits = p['closing_credits']
    
    loan_duration_years = p['loan_duration_years']
    base_interest_rate = p['base_interest_rate']
    points_purchased = p['points_purchased']
    cost_per_point_percent = p['cost_per_point_percent']
    rate_reduction_per_point = p['rate_reduction_per_point']
    
    prop_tax_rate = p['prop_tax_rate']
    hoa_monthly = p['hoa_monthly']
    home_ins_monthly = p['home_ins_monthly']
    pmi_monthly = p['pmi_monthly']
    prop_management_percent = p['prop_management_percent']
    safety_deposit_months = p['safety_deposit_months']
    
    holding_years = p['holding_years']
    appreciation_yoy = p['appreciation_yoy']
    realtor_commission_factor = p['realtor_commission_factor']
    tax_bracket = p['tax_bracket']
    state = p['state']
    
    long_term_cap_gains_rate = p['long_term_cap_gains_rate']
    depreciation_years = p['depreciation_years']
    building_value_ratio = p['building_value_ratio']
    recapture_tax_rate = p['recapture_tax_rate']
    
    sp500_annual_return = p['sp500_annual_return']

    # State Tax Logic
    state_tax_data = {
        'CA': {'income': 0.093, 'cap_gains_map': 'income'},
        'NY': {'income': 0.0633, 'cap_gains_map': 'income'},
        'TX': {'income': 0.00, 'cap_gains_map': 'income'},
        'FL': {'income': 0.00, 'cap_gains_map': 'income'},
        'WA': {'income': 0.00, 'cap_gains_map': 0.07},
        'IL': {'income': 0.0495, 'cap_gains_map': 'income'},
        'MA': {'income': 0.05, 'cap_gains_map': 0.12},
        'PA': {'income': 0.0307, 'cap_gains_map': 'income'},
        'AR': {'income': 0.044, 'cap_gains_map': 0.022},    # Top rate ~4.4%, 50% exclusion on LT Cap Gains
        'None': {'income': 0.00, 'cap_gains_map': 0.00}
    }
    
    state_data = state_tax_data.get(state, {'income': 0.0, 'cap_gains_map': 0.0})
    state_income_rate = state_data['income']
    
    if state_data['cap_gains_map'] == 'income':
        state_cap_gains_rate = state_income_rate
    else:
        state_cap_gains_rate = float(state_data['cap_gains_map'])
        
    combined_ordinary_rate = tax_bracket + state_income_rate

    # ==========================================
    # 2. CALCULATIONS
    # ==========================================

    # Determine Appreciation Basis (Fair Market Value)
    fair_market_value = p.get('fair_market_value')
    if fair_market_value is None:
        fair_market_value = quote_price
    
    # Rent Estimates
    monthly_rent = sqft * rent_per_sqft 
    
    # Loan Amount
    loan_amount = quote_price * (1 - down_percent)
    
    # Mortgage Points
    effective_interest_rate = base_interest_rate - (points_purchased * rate_reduction_per_point)
    points_cost = loan_amount * (points_purchased * cost_per_point_percent / 100)
    
    # Mortgage Payment (EMI)
    monthly_rate = effective_interest_rate / 12
    total_months = loan_duration_years * 12
    
    mortgage_type = p.get('mortgage_type', 'amortizing')
    
    if mortgage_type == 'interest_only':
        emi = loan_amount * monthly_rate
    else:
        # Standard Amortization
        if monthly_rate > 0:
            emi = loan_amount * monthly_rate * (math.pow(1 + monthly_rate, total_months)) / (math.pow(1 + monthly_rate, total_months) - 1)
        else:
            emi = loan_amount / total_months

    # Monthly Expenses
    prop_tax_monthly = (quote_price * prop_tax_rate) / 12
    prop_management_fee = monthly_rent * prop_management_percent
    
    # Total Monthly Payment (PITI + HOA + PMI + Mgmt Fee)
    total_monthly_payment = emi + prop_tax_monthly + home_ins_monthly + hoa_monthly + pmi_monthly + prop_management_fee 
    
    # Investment Cash Requirements
    total_down_amount = quote_price * down_percent
    down_cash_due = total_down_amount - advance_payment 
    safety_reserve = total_monthly_payment * safety_deposit_months
    
    total_starting_investment = (
        down_cash_due + 
        advance_payment + 
        one_time_other + 
        points_cost + 
        safety_reserve - 
        closing_credits
    )

    # Appreciation Rates (defaults if not provided)
    rent_appreciation_yoy = p.get('rent_appreciation_yoy', 0.03)
    expense_appreciation_yoy = p.get('expense_appreciation_yoy', 0.02)
    
    # Precise Monthly Rate for S&P (Effective Annual Rate to Monthly)
    # R_monthly = (1 + R_annual)^(1/12) - 1
    sp500_monthly_rate = math.pow(1 + sp500_annual_return, 1/12) - 1
    
    # ==========================================
    # 3. SIMULATION LOOP (Monthly)
    # ==========================================
    remaining_balance = loan_amount
    total_interest_paid = 0
    total_prop_tax_paid = 0
    total_rent_income = 0
    total_operating_expenses = 0
    
    # House Side Investment Account (Reinvesting monthly profits)
    house_reinvestment_balance = 0
    
    # S&P Side (Starting from same initial capital)
    sp500_balance = total_starting_investment
    sp500_total_invested = total_starting_investment
    
    months_owned = int(holding_years * 12)
    
    for month in range(1, months_owned + 1):
        # Determine Current Year for appreciation
        year = math.ceil(month / 12) - 1
        
        # Grow components
        current_rent = monthly_rent * math.pow(1 + rent_appreciation_yoy, year)
        current_tax = prop_tax_monthly * math.pow(1 + expense_appreciation_yoy, year)
        current_ins = home_ins_monthly * math.pow(1 + expense_appreciation_yoy, year)
        current_hoa = hoa_monthly * math.pow(1 + expense_appreciation_yoy, year)
        current_mgmt = current_rent * prop_management_percent
        
        # Loan Math
        interest_payment = remaining_balance * monthly_rate
        
        # If interest_only, emi logic is handled
        if mortgage_type == 'interest_only':
            actual_emi = interest_payment
            principal_payment = 0
        else:
            actual_emi = emi
            principal_payment = actual_emi - interest_payment
            
        remaining_balance -= principal_payment
        
        # Accumulate Totals
        total_interest_paid += interest_payment
        total_prop_tax_paid += current_tax
        total_rent_income += current_rent
        
        monthly_op_ex = current_tax + current_ins + current_hoa + pmi_monthly + current_mgmt
        total_operating_expenses += monthly_op_ex
        
        # House Cash Flow & Reinvestment
        monthly_out_of_pocket_current = (actual_emi + monthly_op_ex) - current_rent
        if month == 1:
            monthly_out_of_pocket = monthly_out_of_pocket_current
        
        # S&P Logic: If you didn't buy the house, you invest the 'out of pocket' cost every month
        sp500_balance *= (1 + sp500_monthly_rate)
        if monthly_out_of_pocket_current > 0:
            sp500_balance += monthly_out_of_pocket_current
            sp500_total_invested += monthly_out_of_pocket_current
            
        # House Logic: Handle positive cashflow
        house_reinvestment_balance *= (1 + sp500_monthly_rate)
        if monthly_out_of_pocket_current < 0:
            extra_cash = abs(monthly_out_of_pocket_current)
            if p.get('positive_cashflow_strategy', 'reinvest') == 'pay_down_loan':
                remaining_balance = max(0, remaining_balance - extra_cash)
            else:
                house_reinvestment_balance += extra_cash

    # ==========================================
    # 4. EXIT CALCULATIONS
    # ==========================================
    
    # Future Value based on Appreciation
    future_value = fair_market_value * math.pow(1 + appreciation_yoy, holding_years)
    sale_net_price = future_value * realtor_commission_factor
    
    # Depreciation
    building_value = quote_price * building_value_ratio
    annual_depreciation = building_value / depreciation_years
    total_depreciation = annual_depreciation * holding_years
    
    # Capital Gains Tax
    initial_cost_basis = quote_price + one_time_other
    adjusted_cost_basis = initial_cost_basis - total_depreciation
    taxable_gain = sale_net_price - adjusted_cost_basis
    
    capital_gains_tax = 0.0
    if taxable_gain > 0:
        if holding_years <= 1.0:
            capital_gains_tax = taxable_gain * combined_ordinary_rate
        else:
            recapture_portion = min(total_depreciation, taxable_gain)
            recapture_tax_total_rate = recapture_tax_rate + state_income_rate
            tax_on_recapture = recapture_portion * recapture_tax_total_rate
            
            pure_capital_gain = max(0, taxable_gain - recapture_portion)
            long_term_total_rate = long_term_cap_gains_rate + state_cap_gains_rate
            tax_on_capital_gain = pure_capital_gain * long_term_total_rate
            
            capital_gains_tax = tax_on_recapture + tax_on_capital_gain

    # Rental Income Tax
    total_non_principal_expenses = total_interest_paid + total_operating_expenses
    total_taxable_rental_income = total_rent_income - total_non_principal_expenses - total_depreciation
    rental_tax_impact = total_taxable_rental_income * combined_ordinary_rate
    
    # Final Cash At Exit (House side)
    # Includes: Sale Proceeds - Loan + Reinvested Profits + Safety Reserve - Taxes
    cash_from_sale = (sale_net_price - remaining_balance) - capital_gains_tax + safety_reserve
    
    # ==========================================
    # 5. RETURNS ANALYSIS
    # ==========================================
    # Total investment = Starting + contributions over time
    total_invested_denominator = sp500_total_invested 
    
    # End Value of House side = cash_from_sale + house_reinvestment_balance - rental_tax_impact
    ending_value = cash_from_sale + house_reinvestment_balance - rental_tax_impact
    total_net_profit = ending_value - total_invested_denominator
    
    annualized_return = 0.0
    if total_invested_denominator > 0 and ending_value > 0:
        annualized_return = (math.pow(ending_value / total_invested_denominator, 1 / holding_years) - 1) * 100
        
    roi = 0.0
    if total_invested_denominator > 0:
        roi = (total_net_profit / total_invested_denominator) * 100
    
    # S&P Final
    sp500_gross_gain = sp500_balance - sp500_total_invested
    if holding_years > 1:
        sp500_tax_rate = long_term_cap_gains_rate + state_cap_gains_rate
    else:
        sp500_tax_rate = combined_ordinary_rate
        
    sp500_tax = max(0, sp500_gross_gain) * sp500_tax_rate
    sp500_net_profit = sp500_gross_gain - sp500_tax
    
    sp500_roi = 0.0
    if sp500_total_invested > 0:
        sp500_roi = (sp500_net_profit / sp500_total_invested) * 100
        
    sp500_ending_value = sp500_total_invested + sp500_net_profit
    sp500_cagr = (math.pow(sp500_ending_value / sp500_total_invested, 1 / holding_years) - 1) * 100

    net_monthly_pocket = total_net_profit / months_owned
    
    # Return everything in local scope
    return locals()

def print_detailed_report(results):
    """Prints a detailed report for a single deal result dictionary."""
    r = results # short alias
    
    print("\n")
    print("=" * 70)
    print(f" REPORT FOR: {r['p']['deal_name']}")
    print("=" * 70)
    print(f"{' Net Monthly money to pocket':<30} {r['net_monthly_pocket']:,.2f}")
    print(f"{' CAGR':<30} {r['annualized_return']:.2f}%")
    print("=" * 70)
    
    print(f"{'--- PROPERTY & LOAN DETAILS ---':^70}")
    print(f"{'Quote Price':<25} ${r['quote_price']:,.2f}")
    if r['fair_market_value'] != r['quote_price']:
        print(f"{'Fair Market Value':<25} ${r['fair_market_value']:,.2f} (!)")
    print(f"{'Sqft':<25} {r['sqft']}")
    print(f"{'Price $/sqft':<25} ${r['quote_price']/r['sqft']:,.2f}")
    print(f"{'Down Payment %':<25} {r['down_percent']*100:.1f}%")
    print(f"{'Down Payment $':<25} ${r['total_down_amount']:,.2f}")
    print(f"{'Loan Amount':<25} ${r['loan_amount']:,.2f}")
    print(f"{'Interest Rate':<25} {r['effective_interest_rate']*100:.3f}%")
    print(f"{'Points':<25} {r['points_purchased']} points (${r['points_cost']:,.2f})")
    print(f"{'Loan Duration':<25} {r['loan_duration_years']} Years ({r['total_months']} months)")
    print(f"{'Cashflow Strategy':<25} {r['p'].get('positive_cashflow_strategy', 'reinvest')}")
    print("-" * 70)
    
    print(f"{'--- MONTHLY BREAKDOWN ---':^70}")
    print(f"{'Monthly Rent':<25} ${r['monthly_rent']:,.2f}  (${r['rent_per_sqft']:.2f}/sqft)")
    print(f"{'Total Monthly Payment':<25} ${r['total_monthly_payment']:,.2f}")
    print(f"  - EMI (P&I):{'':<15} ${r['emi']:,.2f}")
    print(f"  - Property Tax:{'':<15} ${r['prop_tax_monthly']:,.2f}")
    print(f"  - Insurance:{'':<15} ${r['home_ins_monthly']:,.2f}")
    print(f"  - HOA:{'':<15} ${r['hoa_monthly']:,.2f}")
    print(f"  - PMI:{'':<15} ${r['pmi_monthly']:,.2f}")
    print(f"  - Prop Mgmt ({r['prop_management_percent']*100:.0f}%):{'':<7} ${r['prop_management_fee']:,.2f}")
    print(f"{'Monthly Out of Pocket':<25} ${r['monthly_out_of_pocket']:,.2f}")
    print("-" * 70)

    print(f"{'--- INVESTMENT CASH FLOW ---':^70}")
    print(f"{'Down Payment Due':<25} ${r['down_cash_due']:,.2f}")
    print(f"{'Advance Payment':<25} ${r['advance_payment']:,.2f}")
    print(f"{'One Time Other':<25} ${r['one_time_other']:,.2f}")
    print(f"{'Points Cost':<25} ${r['points_cost']:,.2f}")
    print(f"{'Safety Reserve':<25} ${r['safety_reserve']:,.2f} ({r['safety_deposit_months']} months)")
    print(f"{'Closing Credits':<25} ${r['closing_credits']:,.2f}")
    print(f"{'TOTAL STARTING INVESTMENT':<25} ${r['total_starting_investment']:,.2f}")
    print("-" * 70)

    exit_title = f"--- EXIT PROJECTIONS (After {r['holding_years']} Years) ---"
    print(f"{exit_title:^70}")
    print(f"{'Future Home Value':<25} ${r['future_value']:,.2f} ({r['appreciation_yoy']*100:.1f}% YoY)")
    print(f"{'Loan Balance':<25} ${r['remaining_balance']:,.2f}")
    
    percent_owned = ((r['quote_price'] - r['remaining_balance']) / r['quote_price']) * 100
    print(f"{'Percent of Prop Owned':<25} {percent_owned:.2f}%")
    
    print(f"{'Equity':<25} ${r['future_value'] - r['remaining_balance']:,.2f}")
    print(f"{'Cost of Sale (Comm)':<25} ${r['future_value'] - r['sale_net_price']:,.2f} ({(1-r['realtor_commission_factor'])*100:.1f}%)")
    print(f"{'Net Sale Proceeds':<25} ${r['sale_net_price']:,.2f}")
    print(f"{'State Selected':<25} {r['state']} (Inc: {r['state_income_rate']*100:.2f}%, CG: {r['state_cap_gains_rate']*100:.2f}%)")
    print(f"{'Adjusted Cost Basis':<25} ${r['adjusted_cost_basis']:,.2f} (Includes -${r['total_depreciation']:,.0f} Depr)")
    print(f"{'Taxable Gain':<25} ${r['taxable_gain']:,.2f}")
    print(f"{'Total Tax on Gain':<25} ${r['capital_gains_tax']:,.2f} (Fed+State)")
    print(f"{'CASH FROM SALE (Net)':<25} ${r['cash_from_sale']:,.2f}")
    print("-" * 70)

    print(f"{'--- RETURNS ANALYSIS ---':^70}")
    if r['rental_tax_impact'] < 0:
        print(f"{'Tax Savings (deductions)':<25} ${-r['rental_tax_impact']:,.2f}")
    else:
        print(f"{'Tax Due (Rental Income)':<25} ${r['rental_tax_impact']:,.2f}")
        
    print(f"{'Reinvested Cash Flow':<25} ${r['house_reinvestment_balance']:,.2f}")
    print(f"{'Total Net Profit':<25} ${r['total_net_profit']:,.2f}")
    print(f"{'Total ROI':<25} {r['roi']:.2f}%")
    print(f"{'Annualized Return (CAGR)':<25} {r['annualized_return']:.2f}%")
    print("-" * 70)
    
    print(f"{'--- S&P 500 ALTERNATIVE ---':^70}")
    print(f"{'S&P 500 Annual Return':<25} {r['sp500_annual_return']*100:.1f}%")
    print(f"{'Total Invested':<25} ${r['sp500_total_invested']:,.2f}")
    print(f"{'Ending Balance (Pre-tax)':<25} ${r['sp500_balance']:,.2f}")
    print(f"{'Capital Gains Tax':<25} ${r['sp500_tax']:,.2f}")
    print(f"{'Net Profit (After Tax)':<25} ${r['sp500_net_profit']:,.2f}")
    print(f"{'S&P ROI':<25} {r['sp500_roi']:.2f}%")
    print(f"{'S&P CAGR':<25} {r['sp500_cagr']:.2f}%")
    print("-" * 70)
    
    print(f"{'--- COMPARISON ---':^70}")
    diff = r['total_net_profit'] - r['sp500_net_profit']
    winner = "Rental Property" if diff > 0 else "S&P 500"
    print(f"{'Net Profit Difference':<25} ${abs(diff):,.2f} (Favors {winner})")
    print("=" * 70)


def print_comparison_table(all_results):
    print("\n\n")
    print("=" * 150)
    print(f"{'COMPREHENSIVE COMPARISON TABLE':^150}")
    print("=" * 150)
    
    # 1. Print Header Row with Deal Names
    print(f"{'Metric':<35} |", end="")
    for r in all_results:
        # Truncate name to 20 chars for clean table
        name = r['p']['deal_name'][:20]
        print(f" {name:<20} |", end="")
    print("\n" + "-" * (37 + 23 * len(all_results)))
    
    # 2. Define ALL rows to compare (organized by category)
    metrics = [
        # Property & Loan Details
        ('--- PROPERTY & LOAN ---', None, None),
        ('Quote Price', 'quote_price', '${:,.0f}'),
        ('Sqft', 'sqft', '{:,.0f}'),
        ('Price $/sqft', None, None),  # Calculated dynamically
        ('Down Payment %', 'down_percent', '{:.1%}'),
        ('Down Payment $', 'total_down_amount', '${:,.0f}'),
        ('Loan Amount', 'loan_amount', '${:,.0f}'),
        ('Interest Rate', 'effective_interest_rate', '{:.3%}'),
        ('Loan Duration', 'total_months', '{:.0f} months'),
        ('Mortgage Type', 'mortgage_type', '{}'),
        ('Cashflow Strategy', 'positive_cashflow_strategy', '{}'),
        
        # Monthly Breakdown
        ('--- MONTHLY BREAKDOWN ---', None, None),
        ('Monthly Rent', 'monthly_rent', '${:,.2f}'),
        ('Total Monthly Payment', 'total_monthly_payment', '${:,.2f}'),
        ('  - EMI (P&I)', 'emi', '${:,.2f}'),
        ('  - Property Tax', 'prop_tax_monthly', '${:,.2f}'),
        ('  - Insurance', 'home_ins_monthly', '${:,.2f}'),
        ('  - HOA', 'hoa_monthly', '${:,.2f}'),
        ('  - PMI', 'pmi_monthly', '${:,.2f}'),
        ('  - Prop Mgmt', 'prop_management_fee', '${:,.2f}'),
        ('Monthly Out of Pocket', 'monthly_out_of_pocket', '${:,.2f}'),
        
        # Investment Cashflow
        ('--- INVESTMENT CASHFLOW ---', None, None),
        ('Down Payment Due', 'down_cash_due', '${:,.2f}'),
        ('Advance Payment', 'advance_payment', '${:,.2f}'),
        ('One Time Other', 'one_time_other', '${:,.2f}'),
        ('Points Cost', 'points_cost', '${:,.2f}'),
        ('Safety Reserve', 'safety_reserve', '${:,.2f}'),
        ('Closing Credits', 'closing_credits', '${:,.2f}'),
        ('Total Starting Investment', 'total_starting_investment', '${:,.2f}'),
        
        # Exit Projections
        ('--- EXIT PROJECTIONS ---', None, None),
        ('Future Home Value', 'future_value', '${:,.2f}'),
        ('Loan Balance at Exit', 'remaining_balance', '${:,.2f}'),
        ('Equity', None, None),  # Calculated dynamically
        ('Cost of Sale (Comm)', None, None),  # Calculated dynamically
        ('Net Sale Proceeds', 'sale_net_price', '${:,.2f}'),
        ('Adjusted Cost Basis', 'adjusted_cost_basis', '${:,.2f}'),
        ('Total Depreciation', 'total_depreciation', '${:,.2f}'),
        ('Taxable Gain', 'taxable_gain', '${:,.2f}'),
        ('Capital Gains Tax', 'capital_gains_tax', '${:,.2f}'),
        ('Cash from Sale (Net)', 'cash_from_sale', '${:,.2f}'),
        
        # Returns Analysis
        ('--- RETURNS ANALYSIS ---', None, None),
        ('Rental Tax Impact', 'rental_tax_impact', '${:,.2f}'),
        ('Reinvested Cash Flow', 'house_reinvestment_balance', '${:,.2f}'),
        ('Total Net Profit', 'total_net_profit', '${:,.2f}'),
        ('Total ROI', 'roi', '{:.2f}%'),
        ('Annualized Return (CAGR)', 'annualized_return', '{:.2f}%'),
        
        # S&P 500 Comparison
        ('--- S&P 500 ALTERNATIVE ---', None, None),
        ('S&P Annual Return %', 'sp500_annual_return', '{:.1%}'),
        ('S&P Gross Gain', None, None),  # Calculated dynamically
        ('S&P Tax', None, None),  # Calculated dynamically
        ('S&P Net Profit (After Tax)', 'sp500_net_profit', '${:,.2f}'),
        ('S&P ROI', 'sp500_roi', '{:.2f}%'),
        ('S&P CAGR', 'sp500_cagr', '{:.2f}%'),
        
        # Final Comparison
        ('--- FINAL COMPARISON ---', None, None),
        ('Net Profit Difference', 'profit_difference', '${:,.2f}'),
        ('Winner', 'winner', '{}'),
    ]
    
    # 3. Print Data Rows
    for label, key, fmt in metrics:
        # Section headers
        if label.startswith('---'):
            print(label)
            print("-" * (37 + 23 * len(all_results)))
            continue
        
        # Special handling for dynamically calculated fields
        if label == 'Price $/sqft':
            print(f"{label:<35} |", end="")
            for r in all_results:
                val = r['quote_price'] / r['sqft']
                print(f" ${val:<19.2f} |", end="")
            print()
        elif label == 'Equity':
            print(f"{label:<35} |", end="")
            for r in all_results:
                equity = r['future_value'] - r['remaining_balance']
                print(f" ${equity:<19,.2f} |", end="")
            print()
        elif label == 'Cost of Sale (Comm)':
            print(f"{label:<35} |", end="")
            for r in all_results:
                cost = r['future_value'] - r['sale_net_price']
                print(f" ${cost:<19,.2f} |", end="")
            print()
        elif label == 'S&P Gross Gain':
            print(f"{label:<35} |", end="")
            for r in all_results:
                gross_gain = r['sp500_gross_gain']
                print(f" ${gross_gain:<19,.2f} |", end="")
            print()
        elif label == 'S&P Tax':
            print(f"{label:<35} |", end="")
            for r in all_results:
                sp500_tax = r['sp500_tax']
                print(f" ${sp500_tax:<19,.2f} |", end="")
            print()
        else:
            # Regular fields
            print(f"{label:<35} |", end="")
            for r in all_results:
                
                # Handle special cases
                if key == 'monthly_out_of_pocket':
                    val = -r[key]  # Flip sign for display
                elif key == 'profit_difference':
                    val = r['total_net_profit'] - r['sp500_net_profit']
                elif key == 'winner':
                    diff = r['total_net_profit'] - r['sp500_net_profit']
                    val = "Rental" if diff > 0 else "S&P 500"
                else:
                    val = r.get(key, 'N/A')
                
                if isinstance(val, str):
                    print(f" {val:<20} |", end="")
                else:
                    print(f" {fmt.format(val):<20} |", end="")
            print()
    print("=" * 150)

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION SECTION: DEFINE YOUR DEALS
    # =========================================================================
    # You can copy and paste the blocks below to create new comparisons.
    # All parameters are exposed here for full control.

    # -------------------------------------------------------------------------
    # DEAL 1: Irenturent PA
    # -------------------------------------------------------------------------
    deal1 = {
        'deal_name': 'Irent',
        
        # Property Details
        'sqft': 2300,
        'quote_price': 500000.00,       # Purchase Price
        'fair_market_value': 500000.00, # Instant Equity!
        'rent_per_sqft': 1.31,
        
        # Expenses & Reserves
        'prop_tax_rate': 0.0151,        # 1.51%
        'hoa_monthly': 119.00,
        'home_ins_monthly': 60.00,
        'prop_management_percent': 0.0, # 0% if self-managed
        'safety_deposit_months': 2,     # Reserve fund in months of total costs
        
        # Growth & Taxes
        'holding_years': 3,             # Global default holding period
        'appreciation_yoy': 0.03,       # 3%
        'realtor_commission_factor': 0.94, # 94% net after 6% comm
        'tax_bracket': 0.24,            # Federal Ordinary Income
        'state': 'PA',
        
        # Investment Tax Details (Advanced)
        'long_term_cap_gains_rate': 0.15,
        'depreciation_years': 27.5,
        'building_value_ratio': 0.80,
        'recapture_tax_rate': 0.25,
        
        # S&P 500 Comparison
        'sp500_annual_return': 0.10,
        
        # Self-Contained Loan Scenarios (Financing Details)
        'loan_scenarios': [
            {
                'name': 'Rama', 
                'mortgage_type': 'amortizing', 
                'down_percent': 0.20,
                'base_interest_rate': 0.04,
                'loan_duration_years': 30,
                'advance_payment': 5000.00,
                'one_time_other': 20000.00,
                'closing_credits': 0.00,
                'points_purchased': 0,
                'pmi_monthly': 0.00
            },
            {
                'name': 'SBLOC_paydown', 
                'mortgage_type': 'interest_only', 
                'down_percent': 0.0, 
                'base_interest_rate': 0.045,
                'advance_payment': 5000.00,
                'one_time_other': 20000.00,
                'closing_credits': 0.00,
                'positive_cashflow_strategy': 'pay_down_loan'
            },
            {
                'name': 'SBLOC_paydown_20%', 
                'mortgage_type': 'interest_only', 
                'down_percent': 0.20, 
                'base_interest_rate': 0.045,
                'advance_payment': 5000.00,
                'one_time_other': 20000.00,
                'closing_credits': 0.00,
                'positive_cashflow_strategy': 'pay_down_loan'
            },
            {
                'name': 'SBLOC 5Y', 
                'mortgage_type': 'interest_only', 
                'down_percent': 0.0, 
                'base_interest_rate': 0.045,
                'advance_payment': 5000.00,
                'one_time_other': 20000.00,
                'closing_credits': 0.00
            },
            {
                'name': 'SBLOC 5Y_20%', 
                'mortgage_type': 'interest_only', 
                'down_percent': 0.20, 
                'base_interest_rate': 0.045,
                'advance_payment': 5000.00,
                'one_time_other': 20000.00,
                'closing_credits': 0.00
            },
            {
                'name': 'Morg', 
                'mortgage_type': 'amortizing', 
                'down_percent': 0.0,
                'base_interest_rate': 0.058,
                'loan_duration_years': 30,
                'advance_payment': 5000.00,
                'one_time_other': 20000.00,
                'closing_credits': 0.00,
                'points_purchased': 0,
                'pmi_monthly': 0.00
            },
                       
        ]
    }

    # -------------------------------------------------------------------------
    # DEAL 2: DRhorton Unit
    # -------------------------------------------------------------------------
    deal2 = {
        'deal_name': 'DR_horton_unit',
        
        # Property Details
        'sqft': 2035,
        'quote_price': 370000.00,
        'fair_market_value': 390000.00,
        'rent_per_sqft': 1.0,
        
        # Expenses & Reserves
        'prop_tax_rate': 0.009,
        'hoa_monthly': 29.167,           # Lower HOA
        'home_ins_monthly': 125.00,
        'prop_management_percent': 0.007,
        'safety_deposit_months': 2,
        
        # Growth & Taxes
        'holding_years': 5,
        'appreciation_yoy': 0.03,
        'realtor_commission_factor': 0.94,
        'tax_bracket': 0.24,
        'state': 'AR',
        
        # Investment Tax Details
        'long_term_cap_gains_rate': 0.15,
        'depreciation_years': 27.5,
        'building_value_ratio': 0.80,
        'recapture_tax_rate': 0.25,
        
        # S&P 500 Comparison
        'sp500_annual_return': 0.10,
        
        'loan_scenarios': [
            {
                'name': 'SBLOC 5Y', 
                'mortgage_type': 'interest_only', 
                'down_percent': 0.0, 
                'base_interest_rate': 0.045,
                'advance_payment': 5000.00,
                'one_time_other': 20000.00,
                'closing_credits': 0.00
            },
            {
                'name': 'SBLOC 5Y_20%', 
                'mortgage_type': 'interest_only', 
                'down_percent': 0.20, 
                'base_interest_rate': 0.045,
                'advance_payment': 5000.00,
                'one_time_other': 20000.00,
                'closing_credits': 0.00
            },
            {
                'name': 'Morg_30Y_1', 
                'mortgage_type': 'amortizing', 
                'down_percent': 0.0,
                'base_interest_rate': 0.058,
                'loan_duration_years': 30,
                'advance_payment': 5000.00,
                'one_time_other': 20000.00,
                'closing_credits': 0.00,
                'points_purchased': 0,
                'pmi_monthly': 0.00
            }
        ]
    }

    
    def create_variants(base_deal):
        """
        Generates multiple deal configurations based on the 'loan_scenarios' 
        list provided inside the deal dictionary.
        """
        scenarios = base_deal.get('loan_scenarios')
        
        # If no scenarios, just return the deal as-is (one variant)
        if not scenarios:
            return [base_deal]
            
        variants = []
        for s in scenarios:
            variant = base_deal.copy()
            # Remove the scenarios list from the variant to keep it clean
            if 'loan_scenarios' in variant: 
                del variant['loan_scenarios']
            
            # Update the variant with scenario-specific settings
            variant.update(s)
            
            # Set the final deal name
            opt_name = s.get('name', 'Custom Loan')
            variant['deal_name'] = f"{base_deal['deal_name']} ({opt_name})"
            
            variants.append(variant)
            
        return variants

    # List of all properties to process
    properties = [deal1]
    
    # Generate all mortgage options for each property
    my_deals = []
    for prop in properties:
        my_deals.extend(create_variants(prop))
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    results_list = []
    
    for deal in my_deals:
        # Calculate
        res = calculate_deal(deal)
        results_list.append(res)
        
        # Print Individual Report
        # print_detailed_report(res)
        
    # Print Comparison Table
    print_comparison_table(results_list)