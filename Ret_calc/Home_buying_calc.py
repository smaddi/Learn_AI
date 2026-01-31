import math

def calculate_xirr(cashflows, dates_in_months, guess=0.1, max_iterations=100, tolerance=1e-6):
    """
    Calculate XIRR (Extended Internal Rate of Return) for irregular cashflows.
    
    Args:
        cashflows: List of cash flows (negative = outflow, positive = inflow)
        dates_in_months: List of months when each cashflow occurs (0 = start)
        guess: Initial guess for annual rate
        max_iterations: Maximum iterations for Newton-Raphson
        tolerance: Convergence tolerance
    
    Returns:
        Annual IRR as a decimal (e.g., 0.10 for 10%)
    """
    if not cashflows or len(cashflows) != len(dates_in_months):
        return 0.0
    
    # Convert months to years for annual rate calculation
    dates_in_years = [m / 12.0 for m in dates_in_months]
    
    def npv(rate):
        """Calculate NPV at given annual rate."""
        total = 0.0
        for cf, t in zip(cashflows, dates_in_years):
            if rate <= -1 and t != 0:
                return float('inf')
            total += cf / pow(1 + rate, t)
        return total
    
    def npv_derivative(rate):
        """Calculate derivative of NPV for Newton-Raphson."""
        total = 0.0
        for cf, t in zip(cashflows, dates_in_years):
            if t == 0:
                continue
            if rate <= -1:
                return float('inf')
            total -= t * cf / pow(1 + rate, t + 1)
        return total
    
    # Newton-Raphson method
    rate = guess
    for _ in range(max_iterations):
        npv_val = npv(rate)
        npv_deriv = npv_derivative(rate)
        
        if abs(npv_deriv) < 1e-10:
            break
            
        new_rate = rate - npv_val / npv_deriv
        
        # Bound the rate to reasonable values
        new_rate = max(-0.99, min(new_rate, 10.0))
        
        if abs(new_rate - rate) < tolerance:
            return new_rate
        rate = new_rate
    
    # Fallback: bisection method if Newton-Raphson fails
    low, high = -0.99, 5.0
    for _ in range(100):
        mid = (low + high) / 2
        if npv(mid) > 0:
            low = mid
        else:
            high = mid
        if abs(high - low) < tolerance:
            return mid
    
    return rate


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
        
        # Interest-Only Loan Configuration
        'interest_only_period_years': 3,  # Years of interest-only before amortizing
        
        # Expenses & Reserves
        'prop_tax_rate': 0.0151,        # J9: 1.51%
        'hoa_monthly': 119.00,          # J11
        'home_ins_monthly': 60.00,      # J12
        'pmi_monthly': 0.00,            # J10
        'prop_management_percent': 0.0, # 8% of rent
        'safety_deposit_months': 2,     # Reserve fund in months of rent
        
        # Vacancy & Maintenance
        'vacancy_rate': 0.05,           # 5% vacancy allowance (industry standard 5-8%)
        'maintenance_percent': 0.01,    # 1% of property value annually for maintenance
        'capex_percent': 0.05,          # 5% of rent set aside for CapEx (roof, HVAC, etc.)
        
        # Growth & Taxes
        'holding_years': 5,             # J3: Years to own
        'appreciation_yoy': 0.03,       # J16: 3%
        'rent_appreciation_yoy': 0.03,  # YoY Rent growth
        'expense_appreciation_yoy': 0.02, # YoY Expense growth
        'prop_tax_follows_appreciation': False,  # If True, prop tax grows with home value
        'realtor_commission_factor': 0.94, # J18: 100% - 6% commission = 94% net
        'tax_bracket': 0.24,            # J25: 24% (Federal Ordinary Income)
        
        # State Selection
        'state': 'PA',                  # Options: 'CA', 'TX', 'FL', 'NY', 'WA', 'IL', 'MA', 'PA'
        
        # Investment Tax Details
        'long_term_cap_gains_rate': 0.15, # Federal Long Term
        'depreciation_years': 27.5,       # Residential rental property standard
        'building_value_ratio': 0.80,     # 80% building value (depreciable)
        'recapture_tax_rate': 0.25,       # Depreciation recapture tax rate (Federal)
        'passive_loss_limit': 25000.00,   # Annual passive loss limit against ordinary income
        
        # Exit Strategy Options
        'exit_strategy': 'sell',          # Options: 'sell', '1031_exchange', 'primary_residence'
        'primary_residence_years': 0,     # Years lived in property (for $250k/$500k exclusion)
        'filing_status': 'single',        # Options: 'single', 'married' (affects exclusion amount)
        
        # Refinancing Scenario
        'refinance_enabled': False,       # Whether to model a refinance
        'refinance_year': 3,              # Year to refinance
        'refinance_rate': 0.045,          # New interest rate after refinance
        'refinance_costs_percent': 0.02,  # Closing costs as % of new loan
        'cash_out_amount': 0,             # Cash-out refinance amount (0 = rate-only refi)
        
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
    
    # Validate critical inputs
    if p.get('holding_years', 0) <= 0:
        raise ValueError(f"holding_years must be greater than 0, got: {p.get('holding_years')}")
    
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
    interest_only_period_years = p.get('interest_only_period_years', 3)
    
    prop_tax_rate = p['prop_tax_rate']
    hoa_monthly = p['hoa_monthly']
    home_ins_monthly = p['home_ins_monthly']
    pmi_monthly = p['pmi_monthly']
    prop_management_percent = p['prop_management_percent']
    safety_deposit_months = p['safety_deposit_months']
    
    # Vacancy & Maintenance
    vacancy_rate = p.get('vacancy_rate', 0.05)
    maintenance_percent = p.get('maintenance_percent', 0.01)
    capex_percent = p.get('capex_percent', 0.05)
    
    # Exit Strategy
    exit_strategy = p.get('exit_strategy', 'sell')
    primary_residence_years = p.get('primary_residence_years', 0)
    filing_status = p.get('filing_status', 'single')
    
    # Refinancing
    refinance_enabled = p.get('refinance_enabled', False)
    refinance_year = p.get('refinance_year', 3)
    refinance_rate = p.get('refinance_rate', 0.045)
    refinance_costs_percent = p.get('refinance_costs_percent', 0.02)
    cash_out_amount = p.get('cash_out_amount', 0)
    refinance_month = refinance_year * 12
    
    holding_years = p['holding_years']
    if holding_years <= 0:
        holding_years = 1  # Fallback to minimum 1 year
    
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
    
    # Rent Estimates (gross rent before vacancy)
    monthly_rent_gross = sqft * rent_per_sqft
    monthly_rent = monthly_rent_gross  # Keep for display, effective rent calculated in loop
    
    # Maintenance & CapEx reserves (monthly)
    annual_maintenance = quote_price * maintenance_percent
    monthly_maintenance = annual_maintenance / 12
    monthly_capex_reserve = monthly_rent_gross * capex_percent 
    
    # Loan Amount
    loan_amount = quote_price * (1 - down_percent)
    
    # Mortgage Points
    effective_interest_rate = base_interest_rate - (points_purchased * rate_reduction_per_point)
    points_cost = loan_amount * (points_purchased * cost_per_point_percent / 100)
    
    # Mortgage Payment (EMI)
    monthly_rate = effective_interest_rate / 12
    total_months = loan_duration_years * 12
    
    mortgage_type = p.get('mortgage_type', 'amortizing')
    interest_only_period_months = int(interest_only_period_years * 12)
    
    # Calculate amortizing EMI (used after interest-only period or for fully amortizing loans)
    if monthly_rate > 0:
        emi_amortizing = loan_amount * monthly_rate * (math.pow(1 + monthly_rate, total_months)) / (math.pow(1 + monthly_rate, total_months) - 1)
    else:
        emi_amortizing = loan_amount / total_months
    
    # For interest-only, initial EMI is just interest; will switch to amortizing after period
    if mortgage_type == 'interest_only':
        emi = loan_amount * monthly_rate  # Initial interest-only payment
    else:
        emi = emi_amortizing

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
    prop_tax_follows_appreciation = p.get('prop_tax_follows_appreciation', False)
    passive_loss_limit = p.get('passive_loss_limit', 25000.00)
    
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
    total_rent_income_gross = 0
    total_operating_expenses = 0
    total_vacancy_loss = 0
    total_maintenance_expense = 0
    total_capex_reserve = 0
    
    # House Side Investment Account (Reinvesting monthly profits)
    house_reinvestment_balance = 0
    
    # S&P Side (Starting from same initial capital)
    sp500_balance = total_starting_investment
    sp500_total_invested = total_starting_investment
    
    # Ensure months_owned is at least 1 to avoid empty simulation
    months_owned = max(1, int(holding_years * 12))
    
    # Track monthly out-of-pocket for reporting (first, last, average)
    monthly_out_of_pocket_first = 0
    monthly_out_of_pocket_last = 0
    total_out_of_pocket_sum = 0
    
    # Track cashflows for XIRR calculation
    house_cashflows = [-total_starting_investment]  # Initial investment (outflow)
    house_cashflow_months = [0]  # Month 0
    sp500_cashflows = [-total_starting_investment]  # Initial investment (outflow)
    sp500_cashflow_months = [0]  # Month 0
    
    # Track cumulative cashflow for break-even analysis
    cumulative_cashflow = -total_starting_investment
    break_even_month = None
    monthly_cashflows_list = []  # For break-even tracking
    
    # Track annual cashflows for Cash-on-Cash calculation
    annual_pretax_cashflows = []
    current_year_cashflow = 0
    
    # Refinancing tracking
    refinanced = False
    refinance_costs_paid = 0
    cash_out_received = 0
    current_monthly_rate = monthly_rate
    current_loan_term_remaining = total_months
    
    for month in range(1, months_owned + 1):
        # Determine Current Year for appreciation
        year = math.ceil(month / 12) - 1
        
        # Grow components (gross rent before vacancy)
        current_rent_gross = monthly_rent_gross * math.pow(1 + rent_appreciation_yoy, year)
        
        # Apply vacancy rate to get effective rent
        current_rent = current_rent_gross * (1 - vacancy_rate)
        current_vacancy_loss = current_rent_gross * vacancy_rate
        
        # Maintenance (grows with property value appreciation)
        current_home_value_for_maint = quote_price * math.pow(1 + appreciation_yoy, year)
        current_maintenance = (current_home_value_for_maint * maintenance_percent) / 12
        
        # CapEx reserve (percentage of gross rent)
        current_capex = current_rent_gross * capex_percent
        
        # Property tax: optionally follow home appreciation instead of expense inflation
        if prop_tax_follows_appreciation:
            current_home_value = quote_price * math.pow(1 + appreciation_yoy, year)
            current_tax = (current_home_value * prop_tax_rate) / 12
        else:
            current_tax = prop_tax_monthly * math.pow(1 + expense_appreciation_yoy, year)
        
        current_ins = home_ins_monthly * math.pow(1 + expense_appreciation_yoy, year)
        current_hoa = hoa_monthly * math.pow(1 + expense_appreciation_yoy, year)
        current_mgmt = current_rent * prop_management_percent
        
        # Handle Refinancing
        if refinance_enabled and not refinanced and month == refinance_month:
            refinanced = True
            # Calculate refinance costs
            refinance_costs_paid = remaining_balance * refinance_costs_percent
            
            # Cash-out refinance: increase loan balance
            if cash_out_amount > 0:
                remaining_balance += cash_out_amount
                cash_out_received = cash_out_amount
            
            # Update loan terms
            current_monthly_rate = refinance_rate / 12
            current_loan_term_remaining = total_months - month  # Remaining months on new loan
            
            # Refinance costs come out of pocket this month
            # (Will be added to monthly_out_of_pocket_current below)
        
        # Loan Math - handle interest-only period transition
        effective_monthly_rate = current_monthly_rate if refinanced else monthly_rate
        interest_payment = remaining_balance * effective_monthly_rate
        
        if mortgage_type == 'interest_only' and not refinanced:
            if month <= interest_only_period_months:
                # During interest-only period
                actual_emi = interest_payment
                principal_payment = 0
            else:
                # After interest-only period, switch to amortizing
                # Recalculate EMI based on remaining balance and remaining term
                remaining_months = total_months - month + 1
                if effective_monthly_rate > 0 and remaining_months > 0:
                    actual_emi = remaining_balance * effective_monthly_rate * (math.pow(1 + effective_monthly_rate, remaining_months)) / (math.pow(1 + effective_monthly_rate, remaining_months) - 1)
                else:
                    actual_emi = remaining_balance / max(1, remaining_months)
                principal_payment = actual_emi - interest_payment
        elif refinanced:
            # After refinance, always use amortizing with new rate
            remaining_months = current_loan_term_remaining - (month - refinance_month)
            if effective_monthly_rate > 0 and remaining_months > 0:
                actual_emi = remaining_balance * effective_monthly_rate * (math.pow(1 + effective_monthly_rate, remaining_months)) / (math.pow(1 + effective_monthly_rate, remaining_months) - 1)
            else:
                actual_emi = remaining_balance / max(1, remaining_months)
            principal_payment = actual_emi - interest_payment
        else:
            actual_emi = emi
            principal_payment = actual_emi - interest_payment
            
        remaining_balance -= principal_payment
        remaining_balance = max(0, remaining_balance)  # Prevent negative balance
        
        # Accumulate Totals
        total_interest_paid += interest_payment
        total_prop_tax_paid += current_tax
        total_rent_income += current_rent
        total_rent_income_gross += current_rent_gross
        total_vacancy_loss += current_vacancy_loss
        total_maintenance_expense += current_maintenance
        total_capex_reserve += current_capex
        
        # Operating expenses now include maintenance and capex
        monthly_op_ex = current_tax + current_ins + current_hoa + pmi_monthly + current_mgmt + current_maintenance + current_capex
        total_operating_expenses += monthly_op_ex
        
        # House Cash Flow & Reinvestment
        # Positive = paying out of pocket (expenses > rent)
        # Negative = receiving cash (rent > expenses)
        monthly_out_of_pocket_current = (actual_emi + monthly_op_ex) - current_rent
        
        # Add refinance costs in the refinance month, subtract cash-out received
        if refinance_enabled and month == refinance_month:
            monthly_out_of_pocket_current += refinance_costs_paid
            monthly_out_of_pocket_current -= cash_out_received  # Cash-out reduces out-of-pocket
        
        # Track for break-even and Cash-on-Cash
        monthly_cashflows_list.append(-monthly_out_of_pocket_current)  # Positive = cash received
        cumulative_cashflow += (-monthly_out_of_pocket_current)
        current_year_cashflow += (-monthly_out_of_pocket_current)
        
        # Check for break-even (first month where cumulative becomes positive)
        if break_even_month is None and cumulative_cashflow >= 0:
            break_even_month = month
        
        # Track annual cashflows (reset at year boundary)
        if month % 12 == 0:
            annual_pretax_cashflows.append(current_year_cashflow)
            current_year_cashflow = 0
        
        # Track for reporting
        if month == 1:
            monthly_out_of_pocket_first = monthly_out_of_pocket_current
        monthly_out_of_pocket_last = monthly_out_of_pocket_current
        total_out_of_pocket_sum += monthly_out_of_pocket_current
        
        # S&P Logic: If you didn't buy the house, you invest the 'out of pocket' cost every month
        sp500_balance *= (1 + sp500_monthly_rate)
        if monthly_out_of_pocket_current > 0:
            sp500_balance += monthly_out_of_pocket_current
            sp500_total_invested += monthly_out_of_pocket_current
            # Track for XIRR
            sp500_cashflows.append(-monthly_out_of_pocket_current)  # Outflow
            sp500_cashflow_months.append(month)
            
        # House Logic: Handle positive cashflow
        house_reinvestment_balance *= (1 + sp500_monthly_rate)
        if monthly_out_of_pocket_current < 0:
            extra_cash = abs(monthly_out_of_pocket_current)
            if p.get('positive_cashflow_strategy', 'reinvest') == 'pay_down_loan':
                remaining_balance = max(0, remaining_balance - extra_cash)
            else:
                house_reinvestment_balance += extra_cash
        elif monthly_out_of_pocket_current > 0:
            # Track house contributions for XIRR (when paying out of pocket)
            house_cashflows.append(-monthly_out_of_pocket_current)  # Outflow
            house_cashflow_months.append(month)
    
    # Calculate average monthly out-of-pocket
    monthly_out_of_pocket_avg = total_out_of_pocket_sum / months_owned if months_owned > 0 else 0
    # Keep backward compatibility - use first month value as the main one
    monthly_out_of_pocket = monthly_out_of_pocket_first
    
    # Add any remaining partial year cashflow
    if current_year_cashflow != 0:
        annual_pretax_cashflows.append(current_year_cashflow)
    
    # ==========================================
    # 3.5 KEY INVESTMENT METRICS
    # ==========================================
    
    # Net Operating Income (NOI) - Year 1
    # NOI = Effective Gross Income - Operating Expenses (excluding debt service)
    year1_gross_rent = monthly_rent_gross * 12
    year1_effective_rent = year1_gross_rent * (1 - vacancy_rate)
    year1_operating_expenses = (prop_tax_monthly + home_ins_monthly + hoa_monthly + 
                                 prop_management_fee + monthly_maintenance + monthly_capex_reserve) * 12
    noi_year1 = year1_effective_rent - year1_operating_expenses
    
    # Cap Rate = NOI / Property Value
    cap_rate = (noi_year1 / quote_price * 100) if quote_price > 0 else 0
    
    # Gross Rent Multiplier (GRM) = Price / Annual Gross Rent
    grm = quote_price / year1_gross_rent if year1_gross_rent > 0 else 0
    
    # Debt Service Coverage Ratio (DSCR) = NOI / Annual Debt Service
    annual_debt_service = emi * 12
    dscr = noi_year1 / annual_debt_service if annual_debt_service > 0 else float('inf')
    
    # Cash-on-Cash Return = Annual Pre-tax Cash Flow / Total Cash Invested
    # Using Year 1 cash flow
    year1_cashflow = annual_pretax_cashflows[0] if annual_pretax_cashflows else 0
    cash_on_cash_year1 = (year1_cashflow / total_starting_investment * 100) if total_starting_investment > 0 else 0
    
    # Average annual cash-on-cash over holding period
    avg_annual_cashflow = sum(annual_pretax_cashflows) / len(annual_pretax_cashflows) if annual_pretax_cashflows else 0
    cash_on_cash_avg = (avg_annual_cashflow / total_starting_investment * 100) if total_starting_investment > 0 else 0
    
    # Break-even analysis
    break_even_years = break_even_month / 12 if break_even_month else None

    # ==========================================
    # 3.6 CASHFLOW SOLVERS
    # ==========================================
    
    # 1. Price for Positive Cashflow (Break-even Month 1)
    # Target: Cashflow >= 0 <=> Out_of_pocket <= 0
    # Out_of_pocket = (EMI + OpEx) - EffectiveRent
    # We want EMI + OpEx <= EffectiveRent
    
    # Re-calculate parts that depend on Price to solve for Price
    # Fixed OpEx (Independent of Price)
    # Note: Prop Mgmt Fee depends on Effective Rent, which is fixed here (since Rent is fixed)
    # We must calculate it based on Effective Rent to match the loop logic
    effective_rent = monthly_rent_gross * (1 - vacancy_rate)
    prop_management_fee_effective = effective_rent * prop_management_percent
    
    opex_fixed = home_ins_monthly + hoa_monthly + pmi_monthly + prop_management_fee_effective + monthly_capex_reserve
    
    # Variable OpEx Factor (Per $ of Price)
    # prop_tax = Price * rate/12, maintenance = Price * rate/12
    opex_variable_factor = (prop_tax_rate / 12) + (maintenance_percent / 12)
    
    # Loan EMI Factor (Per $ of Loan Amount)
    # Loan Amount = Price * (1 - down_percent)
    loan_to_price_ratio = (1 - down_percent)
    
    # EMI Factor per $ of Loan (depends on Rate, which is constant here)
    if monthly_rate > 0:
        if mortgage_type == 'interest_only':
            emi_factor_loan = monthly_rate
        else:
            try:
                emi_factor_loan = monthly_rate * math.pow(1 + monthly_rate, total_months) / (math.pow(1 + monthly_rate, total_months) - 1)
            except (OverflowError, ZeroDivisionError):
                 emi_factor_loan = monthly_rate # Fallback
    else:
         if mortgage_type == 'interest_only':
             emi_factor_loan = 0
         else:
             emi_factor_loan = 1 / total_months if total_months > 0 else 0
             
    total_price_factor = opex_variable_factor + (loan_to_price_ratio * emi_factor_loan)
    
    # Effective Rent (Target Revenue) - Month 1
    target_revenue = effective_rent
    
    # Equation: Price * Total_Factor + Fixed_OpEx = Target_Revenue
    # Price = (Target_Revenue - Fixed_OpEx) / Total_Factor
    
    required_price_for_cashflow = 0.0
    if total_price_factor > 0:
        required_price_for_cashflow = (target_revenue - opex_fixed) / total_price_factor
    else:
        required_price_for_cashflow = float('inf') 
        
    # 2. Interest Rate for Positive Cashflow
    # Target: EMI + OpEx <= EffectiveRent
    # EMI <= EffectiveRent - OpEx (at current Price)
    # OpEx includes variable parts at CURRENT Price
    # We reconstruct OpEx to ensure Mgmt Fee is based on Effective Rent
    current_opex_for_solver = prop_tax_monthly + home_ins_monthly + hoa_monthly + pmi_monthly + prop_management_fee_effective + monthly_maintenance + monthly_capex_reserve
    
    max_affordable_emi = target_revenue - current_opex_for_solver
    
    required_interest_rate_pct = 0.0
    
    if max_affordable_emi < 0:
        # Expenses > Rent even without loan
        required_interest_rate_pct = -999.0 # Impossible (Need negative rate)
    elif loan_amount <= 0:
        # No loan, so rate doesn't matter
        required_interest_rate_pct = 0.0 
    else:
        # We need to find rate such that CalculateEMI(rate) = max_affordable_emi
        target_payment_ratio = max_affordable_emi / loan_amount
        
        if mortgage_type == 'interest_only':
            # EMI = Loan * Rate
            # Rate = EMI / Loan
            req_monthly = target_payment_ratio
            required_interest_rate_pct = req_monthly * 12
        else:
            # Solve Amortization: P * r(1+r)^n / ((1+r)^n - 1) = T
            # Monotonic increasing with r. Bisection method.
            
            def amort_payment_factor(r, n):
                if r <= 0: return 1/n if n > 0 else 0
                try:
                    num = r * math.pow(1+r, n)
                    den = math.pow(1+r, n) - 1
                    return num/den
                except OverflowError:
                    return float('inf')

            low_g, high_g = 0.0, 2.0 # 0% to 200% monthly
            # Check feasibility at 0%
            if amort_payment_factor(0, total_months) > target_payment_ratio:
                 required_interest_rate_pct = -1.0 # Even 0% is too expensive (principal payment > affordable)
            else:
                final_rate = 0.0
                for _ in range(50):
                    mid = (low_g + high_g) / 2
                    val = amort_payment_factor(mid, total_months)
                    if abs(val - target_payment_ratio) < 1e-9:
                        final_rate = mid
                        break
                    if val < target_payment_ratio:
                        low_g = mid
                    else:
                        high_g = mid
                
                final_rate = (low_g + high_g) / 2
                required_interest_rate_pct = final_rate * 12

    # ==========================================
    # 4. EXIT CALCULATIONS
    # ==========================================
    
    # Future Value based on Appreciation
    future_value = fair_market_value * math.pow(1 + appreciation_yoy, holding_years)
    sale_net_price = future_value * realtor_commission_factor
    
    # Depreciation - capped at building value
    building_value = quote_price * building_value_ratio
    annual_depreciation = building_value / depreciation_years
    total_depreciation = min(annual_depreciation * holding_years, building_value)
    
    # Capital Gains Tax
    initial_cost_basis = quote_price + one_time_other
    adjusted_cost_basis = initial_cost_basis - total_depreciation
    taxable_gain = sale_net_price - adjusted_cost_basis
    
    # Calculate capital gains tax based on exit strategy
    capital_gains_tax = 0.0
    tax_deferred_1031 = 0.0
    primary_residence_exclusion = 0.0
    
    if exit_strategy == '1031_exchange':
        # 1031 Exchange: Defer ALL capital gains tax (including depreciation recapture)
        # Tax is deferred, not eliminated - track for reporting
        if taxable_gain > 0:
            if holding_years <= 1.0:
                tax_deferred_1031 = taxable_gain * combined_ordinary_rate
            else:
                recapture_portion = min(total_depreciation, taxable_gain)
                recapture_tax_total_rate = recapture_tax_rate + state_income_rate
                tax_on_recapture = recapture_portion * recapture_tax_total_rate
                
                pure_capital_gain = max(0, taxable_gain - recapture_portion)
                long_term_total_rate = long_term_cap_gains_rate + state_cap_gains_rate
                tax_on_capital_gain = pure_capital_gain * long_term_total_rate
                
                tax_deferred_1031 = tax_on_recapture + tax_on_capital_gain
        capital_gains_tax = 0.0  # Deferred
        
    elif exit_strategy == 'primary_residence':
        # Primary Residence Exclusion: Must live in 2 of last 5 years
        # Exclusion: $250k single, $500k married
        if primary_residence_years >= 2 and holding_years <= 5:
            exclusion_amount = 500000.0 if filing_status == 'married' else 250000.0
            
            # Calculate gain (before depreciation recapture considerations)
            # Note: Depreciation recapture still applies for rental period
            pure_appreciation_gain = sale_net_price - initial_cost_basis
            
            # Primary residence exclusion applies to appreciation, not depreciation recapture
            excluded_gain = min(max(0, pure_appreciation_gain), exclusion_amount)
            primary_residence_exclusion = excluded_gain
            
            # Depreciation recapture is NOT excluded - still taxable
            recapture_portion = min(total_depreciation, taxable_gain)
            recapture_tax_total_rate = recapture_tax_rate + state_income_rate
            tax_on_recapture = recapture_portion * recapture_tax_total_rate
            
            # Remaining gain after exclusion
            remaining_gain = max(0, pure_appreciation_gain - excluded_gain)
            if remaining_gain > 0 and holding_years > 1:
                long_term_total_rate = long_term_cap_gains_rate + state_cap_gains_rate
                tax_on_remaining = remaining_gain * long_term_total_rate
            else:
                tax_on_remaining = remaining_gain * combined_ordinary_rate if holding_years <= 1 else 0
            
            capital_gains_tax = tax_on_recapture + tax_on_remaining
        else:
            # Doesn't qualify for exclusion, normal tax calculation
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
    else:
        # Normal sale - standard capital gains calculation
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

    # Rental Income Tax with Passive Loss Limitation
    total_non_principal_expenses = total_interest_paid + total_operating_expenses
    total_taxable_rental_income = total_rent_income - total_non_principal_expenses - total_depreciation
    
    # Apply passive loss limitation (simplified annual calculation)
    # If taxable_rental_income is negative (a loss), limit deduction to passive_loss_limit per year
    if total_taxable_rental_income < 0:
        max_allowed_loss = passive_loss_limit * holding_years
        # The usable loss is the smaller of actual loss or allowed limit
        usable_loss = max(total_taxable_rental_income, -max_allowed_loss)
        suspended_passive_losses = total_taxable_rental_income - usable_loss  # Negative value
        rental_tax_impact = usable_loss * combined_ordinary_rate  # Tax benefit (negative)
    else:
        suspended_passive_losses = 0
        rental_tax_impact = total_taxable_rental_income * combined_ordinary_rate  # Tax owed (positive)
    
    # At sale, suspended passive losses are released and can offset gains
    # (Simplified: we just track them for reporting, already included in rental_tax_impact calculation)
    
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
    
    # Calculate proper XIRR for house investment
    # Add final cashflow (ending value at sale)
    house_cashflows.append(ending_value)
    house_cashflow_months.append(months_owned)
    
    # Calculate annualized return using XIRR
    annualized_return = 0.0
    if total_invested_denominator > 0 and holding_years > 0:
        if ending_value > 0:
            xirr_rate = calculate_xirr(house_cashflows, house_cashflow_months)
            annualized_return = xirr_rate * 100
        else:
            # Handle loss scenario - calculate negative return
            # Use simple formula as approximation for losses
            annualized_return = (math.pow(max(0.001, ending_value) / total_invested_denominator, 1 / holding_years) - 1) * 100
            if ending_value <= 0:
                annualized_return = -100.0  # Complete loss
        
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
    
    # Calculate proper XIRR for S&P 500 investment
    sp500_ending_value = sp500_total_invested + sp500_net_profit
    sp500_cashflows.append(sp500_ending_value)
    sp500_cashflow_months.append(months_owned)
    
    sp500_cagr = 0.0
    if sp500_total_invested > 0:
        if sp500_ending_value > 0:
            sp500_xirr_rate = calculate_xirr(sp500_cashflows, sp500_cashflow_months)
            sp500_cagr = sp500_xirr_rate * 100
        else:
            sp500_cagr = -100.0  # Complete loss

    net_monthly_pocket = total_net_profit / months_owned if months_owned > 0 else 0
    
    # Breakdown of Cash Invested
    total_monthly_contributions = sp500_total_invested - total_starting_investment
    
    # ==========================================
    # 6. BREAK-EVEN PROPERTY APPRECIATION RATES
    # ==========================================
    # Calculate minimum YoY appreciation rate for property to break even on:
    # 1. Total Net Profit = 0
    # 2. Net Profit Difference = 0 (match S&P 500)
    
    def calculate_breakeven_appreciation_rate(target_net_profit):
        """
        Binary search to find the minimum appreciation_yoy rate that achieves the target_net_profit.
        Returns the appreciation rate as a decimal (e.g., 0.033 for 3.3%).
        Returns None if no valid rate exists in reasonable range.
        """
        # Search range: -10% to +20% appreciation
        low, high = -0.10, 0.20
        tolerance = 0.0001  # 0.01% precision
        max_iterations = 100
        
        for iteration in range(max_iterations):
            mid = (low + high) / 2
            
            # Create a temp params dict with modified appreciation
            temp_params = p.copy()
            temp_params['appreciation_yoy'] = mid
            
            # Recalculate just the necessary parts
            # We need: future_value, capital_gains_tax, ending_value, total_net_profit
            temp_future_value = fair_market_value * math.pow(1 + mid, holding_years)
            temp_sale_net_price = temp_future_value * realtor_commission_factor
            
            # Tax calculation (simplified - reuse most existing values)
            temp_taxable_gain = max(0, temp_sale_net_price - adjusted_cost_basis)
            
            # Calculate capital gains tax based on exit strategy
            if exit_strategy == '1031_exchange':
                temp_capital_gains_tax = 0
            elif exit_strategy == 'primary_residence':
                exclusion_amount = 0
                if primary_residence_years >= 2:
                    exclusion_amount = 500000 if filing_status == 'married' else 250000
                
                excluded_gain = min(exclusion_amount, temp_taxable_gain)
                remaining_gain = max(0, temp_taxable_gain - excluded_gain)
                
                if remaining_gain > 0:
                    if holding_years <= 1.0:
                        temp_capital_gains_tax = remaining_gain * combined_ordinary_rate
                    else:
                        recapture_portion = min(total_depreciation, remaining_gain)
                        recapture_tax_total_rate = recapture_tax_rate + state_income_rate
                        tax_on_recapture = recapture_portion * recapture_tax_total_rate
                        
                        pure_capital_gain = max(0, remaining_gain - recapture_portion)
                        long_term_total_rate = long_term_cap_gains_rate + state_cap_gains_rate
                        tax_on_remaining = pure_capital_gain * long_term_total_rate
                        
                        temp_capital_gains_tax = tax_on_recapture + tax_on_remaining
                else:
                    temp_capital_gains_tax = 0
            else:
                if temp_taxable_gain > 0:
                    if holding_years <= 1.0:
                        temp_capital_gains_tax = temp_taxable_gain * combined_ordinary_rate
                    else:
                        recapture_portion = min(total_depreciation, temp_taxable_gain)
                        recapture_tax_total_rate = recapture_tax_rate + state_income_rate
                        tax_on_recapture = recapture_portion * recapture_tax_total_rate
                        
                        pure_capital_gain = max(0, temp_taxable_gain - recapture_portion)
                        long_term_total_rate = long_term_cap_gains_rate + state_cap_gains_rate
                        tax_on_capital_gain = pure_capital_gain * long_term_total_rate
                        
                        temp_capital_gains_tax = tax_on_recapture + tax_on_capital_gain
                else:
                    temp_capital_gains_tax = 0
            
            temp_cash_from_sale = (temp_sale_net_price - remaining_balance) - temp_capital_gains_tax + safety_reserve
            temp_ending_value = temp_cash_from_sale + house_reinvestment_balance - rental_tax_impact
            temp_total_net_profit = temp_ending_value - total_invested_denominator
            
            # Check if we've found the target
            if abs(temp_total_net_profit - target_net_profit) < tolerance:
                return mid
            
            # Adjust search range
            if temp_total_net_profit < target_net_profit:
                low = mid  # Need higher appreciation
            else:
                high = mid  # Need lower appreciation
            
            # Check for convergence
            if high - low < tolerance:
                return mid
        
        # If we didn't converge, return the best estimate
        return mid
    
    # Calculate break-even rates
    breakeven_appreciation_for_zero_profit = calculate_breakeven_appreciation_rate(0)
    breakeven_appreciation_for_sp500_match = calculate_breakeven_appreciation_rate(sp500_net_profit)
    
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
    price_per_sqft = r['quote_price'] / r['sqft'] if r['sqft'] > 0 else 0
    print(f"{'Price $/sqft':<25} ${price_per_sqft:,.2f}")
    print(f"{'Down Payment %':<25} {r['down_percent']*100:.1f}%")
    print(f"{'Down Payment $':<25} ${r['total_down_amount']:,.2f}")
    print(f"{'Loan Amount':<25} ${r['loan_amount']:,.2f}")
    print(f"{'Interest Rate':<25} {r['effective_interest_rate']*100:.3f}%")
    print(f"{'Points':<25} {r['points_purchased']} points (${r['points_cost']:,.2f})")
    mortgage_type = r.get('mortgage_type', 'amortizing')
    if mortgage_type == 'interest_only':
        io_period = r.get('interest_only_period_years', 3)
        print(f"{'Loan Type':<25} Interest-Only ({io_period} yrs) then Amortizing")
    else:
        print(f"{'Loan Type':<25} Amortizing")
    print(f"{'Loan Duration':<25} {r['loan_duration_years']} Years ({r['total_months']} months)")
    print(f"{'Cashflow Strategy':<25} {r['p'].get('positive_cashflow_strategy', 'reinvest')}")
    print("-" * 70)
    
    print(f"{'--- MONTHLY BREAKDOWN ---':^70}")
    print(f"{'Monthly Rent (Gross)':<25} ${r['monthly_rent_gross']:,.2f}  (${r['rent_per_sqft']:.2f}/sqft)")
    print(f"{'Vacancy Rate':<25} {r['vacancy_rate']*100:.1f}%")
    print(f"{'Monthly Rent (Effective)':<25} ${r['monthly_rent_gross']*(1-r['vacancy_rate']):,.2f}")
    print(f"{'Total Monthly Payment':<25} ${r['total_monthly_payment']:,.2f}")
    print(f"  - EMI (P&I):{'':<15} ${r['emi']:,.2f}")
    print(f"  - Property Tax:{'':<15} ${r['prop_tax_monthly']:,.2f}")
    print(f"  - Insurance:{'':<15} ${r['home_ins_monthly']:,.2f}")
    print(f"  - HOA:{'':<15} ${r['hoa_monthly']:,.2f}")
    print(f"  - PMI:{'':<15} ${r['pmi_monthly']:,.2f}")
    print(f"  - Prop Mgmt ({r['prop_management_percent']*100:.0f}%):{'':<7} ${r['prop_management_fee']:,.2f}")
    print(f"  - Maintenance ({r['maintenance_percent']*100:.1f}%):{'':<5} ${r['monthly_maintenance']:,.2f}")
    print(f"  - CapEx ({r['capex_percent']*100:.0f}%):{'':<10} ${r['monthly_capex_reserve']:,.2f}")
    # Monthly out of pocket: positive = paying, negative = receiving cashflow
    oop_first = r.get('monthly_out_of_pocket_first', r['monthly_out_of_pocket'])
    oop_last = r.get('monthly_out_of_pocket_last', r['monthly_out_of_pocket'])
    oop_avg = r.get('monthly_out_of_pocket_avg', r['monthly_out_of_pocket'])
    print(f"{'Out of Pocket (Month 1)':<25} ${oop_first:,.2f}")
    print(f"{'Out of Pocket (Final Mo)':<25} ${oop_last:,.2f}")
    print(f"{'Out of Pocket (Average)':<25} ${oop_avg:,.2f}")
    print("-" * 70)
    
    print(f"{'--- KEY INVESTMENT METRICS ---':^70}")
    print(f"{'NOI (Year 1)':<25} ${r['noi_year1']:,.2f}")
    print(f"{'Cap Rate':<25} {r['cap_rate']:.2f}%")
    print(f"{'Gross Rent Multiplier':<25} {r['grm']:.2f}")
    print(f"{'DSCR':<25} {r['dscr']:.2f}")
    print(f"{'Cash-on-Cash (Year 1)':<25} {r['cash_on_cash_year1']:.2f}%")
    print(f"{'Cash-on-Cash (Average)':<25} {r['cash_on_cash_avg']:.2f}%")
    be_years = r.get('break_even_years')
    if be_years:
        print(f"{'Break-Even':<25} {be_years:.1f} years ({r['break_even_month']} months)")
    else:
        print(f"{'Break-Even':<25} Not reached in holding period")
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
    
    # Equity percentage = (Future Value - Loan Balance) / Future Value
    equity = r['future_value'] - r['remaining_balance']
    percent_equity = (equity / r['future_value'] * 100) if r['future_value'] > 0 else 0
    print(f"{'Equity % of Home Value':<25} {percent_equity:.2f}%")
    
    print(f"{'Equity':<25} ${r['future_value'] - r['remaining_balance']:,.2f}")
    print(f"{'Cost of Sale (Comm)':<25} ${r['future_value'] - r['sale_net_price']:,.2f} ({(1-r['realtor_commission_factor'])*100:.1f}%)")
    print(f"{'Net Sale Proceeds':<25} ${r['sale_net_price']:,.2f}")
    print(f"{'State Selected':<25} {r['state']} (Inc: {r['state_income_rate']*100:.2f}%, CG: {r['state_cap_gains_rate']*100:.2f}%)")
    print(f"{'Adjusted Cost Basis':<25} ${r['adjusted_cost_basis']:,.2f} (Includes -${r['total_depreciation']:,.0f} Depr)")
    print(f"{'Taxable Gain':<25} ${r['taxable_gain']:,.2f}")
    
    # Show exit strategy details
    exit_strat = r.get('exit_strategy', 'sell')
    print(f"{'Exit Strategy':<25} {exit_strat}")
    if exit_strat == '1031_exchange':
        print(f"{'Tax Deferred (1031)':<25} ${r.get('tax_deferred_1031', 0):,.2f}")
        print(f"{'Total Tax on Gain':<25} $0.00 (Deferred)")
    elif exit_strat == 'primary_residence':
        exclusion = r.get('primary_residence_exclusion', 0)
        if exclusion > 0:
            print(f"{'Primary Res Exclusion':<25} ${exclusion:,.2f}")
        print(f"{'Total Tax on Gain':<25} ${r['capital_gains_tax']:,.2f} (Fed+State)")
    else:
        print(f"{'Total Tax on Gain':<25} ${r['capital_gains_tax']:,.2f} (Fed+State)")
    print(f"{'CASH FROM SALE (Net)':<25} ${r['cash_from_sale']:,.2f}")
    print("-" * 70)

    print(f"{'--- RETURNS ANALYSIS ---':^70}")
    if r['rental_tax_impact'] < 0:
        print(f"{'Tax Savings (deductions)':<25} ${-r['rental_tax_impact']:,.2f}")
    else:
        print(f"{'Tax Due (Rental Income)':<25} ${r['rental_tax_impact']:,.2f}")
        
    print(f"{'Reinvested Cash Flow':<25} ${r['house_reinvestment_balance']:,.2f}")
    print(f"{'Total Cash Invested':<25} ${r['sp500_total_invested']:,.2f}")
    print(f"{'  - Initial Investment':<25} ${r['total_starting_investment']:,.2f}")
    print(f"{'  - Recurring Contribs':<25} ${r['total_monthly_contributions']:,.2f}")
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
    table_width = 85 + 23 * len(all_results)
    print("=" * table_width)
    print(f"{'COMPREHENSIVE COMPARISON TABLE':^{table_width}}")
    print("=" * table_width)
    
    # 1. Print Header Row with Deal Names
    print(f"{'Metric':<35} | {'Description':<45} |", end="")
    for r in all_results:
        # Truncate name to 20 chars for clean table
        name = r['p']['deal_name'][:20]
        print(f" {name:<20} |", end="")
    print("\n" + "-" * (85 + 23 * len(all_results)))
    
    # 2. Define ALL rows to compare (organized by category)
    # Format: (label, key, format, description)
    metrics = [
        # Property & Loan Details
        ('--- PROPERTY & LOAN ---', None, None, None),
        ('Quote Price', 'quote_price', '${:,.0f}', 'Purchase price of the property'),
        ('Sqft', 'sqft', '{:,.0f}', 'Total square footage'),
        ('Price $/sqft', None, None, 'Cost per square foot'),
        ('Down Payment %', 'down_percent', '{:.1%}', 'Percentage of price paid upfront'),
        ('Down Payment $', 'total_down_amount', '${:,.0f}', 'Cash paid upfront'),
        ('Loan Amount', 'loan_amount', '${:,.0f}', 'Amount borrowed from lender'),
        ('Interest Rate', 'effective_interest_rate', '{:.3%}', 'Annual loan interest rate'),
        ('Loan Duration', 'total_months', '{:.0f} months', 'Loan repayment period'),
        ('Mortgage Type', 'mortgage_type', '{}', 'amortizing=principal+interest, interest_only=interest only'),
        ('I/O Period (years)', 'interest_only_period_years', '{:.0f}', 'Years of interest-only payments'),
        ('Cashflow Strategy', 'positive_cashflow_strategy', '{}', 'reinvest=invest profits, pay_down_loan=reduce principal'),
        
        # Monthly Breakdown
        ('--- MONTHLY BREAKDOWN ---', None, None, None),
        ('Monthly Rent (Gross)', 'monthly_rent_gross', '${:,.2f}', 'Rent income before vacancy'),
        ('Vacancy Rate', 'vacancy_rate', '{:.1%}', 'Expected % of time property is vacant'),
        ('Monthly Rent (Effective)', None, None, 'Actual rent after vacancy losses'),
        ('Total Monthly Payment', 'total_monthly_payment', '${:,.2f}', 'All monthly expenses combined'),
        ('  - EMI (P&I)', 'emi', '${:,.2f}', 'Monthly loan payment (principal + interest)'),
        ('  - Property Tax', 'prop_tax_monthly', '${:,.2f}', 'Monthly property tax'),
        ('  - Insurance', 'home_ins_monthly', '${:,.2f}', 'Monthly homeowner insurance'),
        ('  - HOA', 'hoa_monthly', '${:,.2f}', 'Monthly HOA fees'),
        ('  - PMI', 'pmi_monthly', '${:,.2f}', 'Private mortgage insurance (if <20% down)'),
        ('  - Prop Mgmt', 'prop_management_fee', '${:,.2f}', 'Property management fee'),
        ('  - Maintenance', 'monthly_maintenance', '${:,.2f}', 'Ongoing repair/upkeep costs'),
        ('  - CapEx Reserve', 'monthly_capex_reserve', '${:,.2f}', 'Savings for major repairs (roof, HVAC)'),
        ('Out of Pocket (Month 1)', 'monthly_out_of_pocket_first', '${:,.2f}', 'Cash needed month 1 (+pay, -receive)'),
        ('Out of Pocket (Final Mo)', 'monthly_out_of_pocket_last', '${:,.2f}', 'Cash needed final month'),
        ('Out of Pocket (Average)', 'monthly_out_of_pocket_avg', '${:,.2f}', 'Avg monthly cash needed over holding'),
        
        # Key Investment Metrics
        ('--- KEY INVESTMENT METRICS ---', None, None, None),
        ('NOI (Year 1)', 'noi_year1', '${:,.2f}', 'Net Operating Income = Rent - Expenses (excl. loan)'),
        ('Cap Rate', 'cap_rate', '{:.2f}%', 'NOI / Price. Higher=better yield. 5-10% typical'),
        ('Gross Rent Multiplier', 'grm', '{:.2f}', 'Price / Annual Rent. Lower=better value'),
        ('DSCR', 'dscr', '{:.2f}', 'NOI / Debt Service. >1.25 is healthy'),
        ('Cash-on-Cash (Year 1)', 'cash_on_cash_year1', '{:.2f}%', 'Year 1 cashflow / Total cash invested'),
        ('Cash-on-Cash (Average)', 'cash_on_cash_avg', '{:.2f}%', 'Avg annual cashflow / Total cash invested'),
        ('Break-Even (months)', 'break_even_month', '{}', 'Months until cumulative cashflow turns positive'),
        
        # Investment Cashflow
        ('--- INVESTMENT CASHFLOW ---', None, None, None),
        ('Down Payment Due', 'down_cash_due', '${:,.2f}', 'Down payment minus advance'),
        ('Advance Payment', 'advance_payment', '${:,.2f}', 'Earnest money / deposit'),
        ('One Time Other', 'one_time_other', '${:,.2f}', 'Closing costs, repairs, etc.'),
        ('Points Cost', 'points_cost', '${:,.2f}', 'Cost to buy down interest rate'),
        ('Safety Reserve', 'safety_reserve', '${:,.2f}', 'Emergency fund (months of expenses)'),
        ('Closing Credits', 'closing_credits', '${:,.2f}', 'Seller/lender credits received'),
        ('Total Starting Investment', 'total_starting_investment', '${:,.2f}', 'Total cash needed to close'),
        
        # Exit Projections
        ('--- EXIT PROJECTIONS ---', None, None, None),
        ('Exit Strategy', 'exit_strategy', '{}', 'sell, 1031_exchange, or primary_residence'),
        ('Future Home Value', 'future_value', '${:,.2f}', 'Projected value at sale'),
        ('Loan Balance at Exit', 'remaining_balance', '${:,.2f}', 'Amount still owed on loan'),
        ('Equity', None, None, 'Home value minus loan balance'),
        ('Equity %', None, None, 'Equity as % of home value'),
        ('Cost of Sale (Comm)', None, None, 'Realtor commission (typically 5-6%)'),
        ('Net Sale Proceeds', 'sale_net_price', '${:,.2f}', 'Sale price after commission'),
        ('Adjusted Cost Basis', 'adjusted_cost_basis', '${:,.2f}', 'Original cost minus depreciation'),
        ('Total Depreciation', 'total_depreciation', '${:,.2f}', 'Tax deduction for building wear'),
        ('Taxable Gain', 'taxable_gain', '${:,.2f}', 'Profit subject to taxes'),
        ('1031 Tax Deferred', 'tax_deferred_1031', '${:,.2f}', 'Tax avoided via 1031 exchange'),
        ('Prim Res Exclusion', 'primary_residence_exclusion', '${:,.2f}', '$250k/$500k exclusion if lived 2+ yrs'),
        ('Capital Gains Tax', 'capital_gains_tax', '${:,.2f}', 'Tax owed on sale profit'),
        ('Cash from Sale (Net)', 'cash_from_sale', '${:,.2f}', 'Cash received after sale & taxes'),
        
        # Returns Analysis
        ('--- RETURNS ANALYSIS ---', None, None, None),
        ('Rental Tax Impact', 'rental_tax_impact', '${:,.2f}', 'Tax owed (+) or saved (-) from rental'),
        ('Reinvested Cash Flow', 'house_reinvestment_balance', '${:,.2f}', 'Positive cashflow invested in S&P'),
        ('Total Cash Invested', 'sp500_total_invested', '${:,.2f}', 'Total cash contributed (Initial + Out-of-pocket)'),
        ('  - Initial Investment', 'total_starting_investment', '${:,.2f}', 'Upfront cash required'),
        ('  - Recurring Contribs', 'total_monthly_contributions', '${:,.2f}', 'Sum of monthly out-of-pocket costs'),
        ('Total Net Profit', 'total_net_profit', '${:,.2f}', 'Final profit after all costs & taxes'),
        ('Total ROI', 'roi', '{:.2f}%', 'Total profit / Total invested'),
        ('Annualized Return (CAGR)', 'annualized_return', '{:.2f}%', 'Compound annual growth rate'),
        
        # S&P 500 Comparison
        ('--- S&P 500 ALTERNATIVE ---', None, None, None),
        ('S&P Annual Return %', 'sp500_annual_return', '{:.1%}', 'Assumed S&P 500 yearly return'),
        ('S&P Gross Gain', None, None, 'Pre-tax profit from S&P'),
        ('S&P Tax', None, None, 'Capital gains tax on S&P profits'),
        ('S&P Net Profit (After Tax)', 'sp500_net_profit', '${:,.2f}', 'S&P profit after taxes'),
        ('S&P ROI', 'sp500_roi', '{:.2f}%', 'S&P profit / Total invested'),
        ('S&P CAGR', 'sp500_cagr', '{:.2f}%', 'S&P compound annual growth rate'),
        
        # Positive Cashflow Requirements
        ('--- POSITIVE FLOW REQ ---', None, None, None),
        ('Price for Pos Cashflow', 'required_price_for_cashflow', '${:,.0f}', 'Max purchase price to break even on M1 cashflow'),
        ('Interest Rate for Pos CF', 'required_interest_rate_pct', '{:.3%}', 'Max interest rate to break even on M1 cashflow'),
        ('Min Apprec Rate (Profit=0)', 'breakeven_appreciation_for_zero_profit', '{:.2%}', 'Min YoY appreciation for Total Net Profit = 0'),
        ('Min Apprec Rate (Match S&P)', 'breakeven_appreciation_for_sp500_match', '{:.2%}', 'Min YoY appreciation to match S&P 500 profit'),
        
        # Final Comparison
        ('--- FINAL COMPARISON ---', None, None, None),
        ('Net Profit Difference', 'profit_difference', '${:,.2f}', 'Rental profit minus S&P profit'),
        ('Winner', 'winner', '{}', 'Which investment performed better'),
    ]
    
    # 3. Print Data Rows
    for label, key, fmt, desc in metrics:
        # Section headers
        if label.startswith('---'):
            print(label)
            print("-" * (85 + 23 * len(all_results)))
            continue
        
        # Get description (truncate to 45 chars)
        description = (desc[:42] + '...') if desc and len(desc) > 45 else (desc or '')
        
        # Special handling for dynamically calculated fields
        if label == 'Price $/sqft':
            print(f"{label:<35} | {description:<45} |", end="")
            for r in all_results:
                val = r['quote_price'] / r['sqft'] if r['sqft'] > 0 else 0
                print(f" ${val:<19.2f} |", end="")
            print()
        elif label == 'Equity':
            print(f"{label:<35} | {description:<45} |", end="")
            for r in all_results:
                equity = r['future_value'] - r['remaining_balance']
                print(f" ${equity:<19,.2f} |", end="")
            print()
        elif label == 'Equity %':
            print(f"{label:<35} | {description:<45} |", end="")
            for r in all_results:
                equity = r['future_value'] - r['remaining_balance']
                percent = (equity / r['future_value'] * 100) if r['future_value'] > 0 else 0
                print(f" {percent:<19.2f}% |", end="")
            print()
        elif label == 'Monthly Rent (Effective)':
            print(f"{label:<35} | {description:<45} |", end="")
            for r in all_results:
                effective = r['monthly_rent_gross'] * (1 - r['vacancy_rate'])
                print(f" ${effective:<18,.2f} |", end="")
            print()
        elif label == 'Cost of Sale (Comm)':
            print(f"{label:<35} | {description:<45} |", end="")
            for r in all_results:
                cost = r['future_value'] - r['sale_net_price']
                print(f" ${cost:<19,.2f} |", end="")
            print()
        elif label == 'S&P Gross Gain':
            print(f"{label:<35} | {description:<45} |", end="")
            for r in all_results:
                gross_gain = r['sp500_gross_gain']
                print(f" ${gross_gain:<19,.2f} |", end="")
            print()
        elif label == 'S&P Tax':
            print(f"{label:<35} | {description:<45} |", end="")
            for r in all_results:
                sp500_tax = r['sp500_tax']
                print(f" ${sp500_tax:<19,.2f} |", end="")
            print()
        else:
            # Regular fields
            print(f"{label:<35} | {description:<45} |", end="")
            for r in all_results:
                
                # Handle special cases
                if key == 'profit_difference':
                    val = r['total_net_profit'] - r['sp500_net_profit']
                elif key == 'winner':
                    diff = r['total_net_profit'] - r['sp500_net_profit']
                    val = "Rental" if diff > 0 else "S&P 500"
                elif key == 'required_price_for_cashflow':
                     val = r.get('required_price_for_cashflow', 0)
                     if val < 0:
                         val = "Impossible (Neg)"
                     elif val == float('inf'):
                         val = "Any Price"
                elif key == 'required_interest_rate_pct':
                     val = r.get('required_interest_rate_pct', 0)
                     if val < 0:
                         val = "Impossible"
                elif key == 'monthly_out_of_pocket_first':
                    val = r.get('monthly_out_of_pocket_first', r.get('monthly_out_of_pocket', 0))
                elif key == 'monthly_out_of_pocket_last':
                    val = r.get('monthly_out_of_pocket_last', r.get('monthly_out_of_pocket', 0))
                elif key == 'monthly_out_of_pocket_avg':
                    val = r.get('monthly_out_of_pocket_avg', r.get('monthly_out_of_pocket', 0))
                elif key == 'positive_cashflow_strategy':
                    val = r.get('p', {}).get('positive_cashflow_strategy', r.get(key, 'reinvest'))
                elif key == 'interest_only_period_years':
                    # Only show I/O period for interest_only mortgages
                    mortgage_type = r.get('mortgage_type', 'amortizing')
                    if mortgage_type == 'interest_only':
                        val = r.get('interest_only_period_years', r.get('p', {}).get('interest_only_period_years', 3))
                    else:
                        val = 'N/A'  # Amortizing loans don't have an I/O period
                elif key == 'break_even_month':
                    val = r.get('break_even_month')
                    if val is None:
                        val = 'N/A'
                elif key == 'exit_strategy':
                    val = r.get('exit_strategy', r.get('p', {}).get('exit_strategy', 'sell'))
                elif key == 'tax_deferred_1031':
                    val = r.get('tax_deferred_1031', 0)
                elif key == 'primary_residence_exclusion':
                    val = r.get('primary_residence_exclusion', 0)
                else:
                    val = r.get(key, 'N/A')
                
                if isinstance(val, str):
                    print(f" {val:<20} |", end="")
                elif val == 'N/A':
                    print(f" {'N/A':<20} |", end="")
                else:
                    print(f" {fmt.format(val):<20} |", end="")
            print()
    print("=" * (85 + 23 * len(all_results)))

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
        'sqft': 2114,
        'quote_price': 570000.00,       # Purchase Price
        'fair_market_value': 588000.00, # Instant Equity!
        'rent_per_sqft': 1.65,
        
        # Expenses & Reserves
        'prop_tax_rate': 0.0130,        # 1.51%
        'hoa_monthly': 220.00,
        'home_ins_monthly': 80.00,
        'prop_management_percent': 0.0, # 0% if self-managed
        'safety_deposit_months': 0,     # Reserve fund in months of total costs
        'capex_percent': 0.00,
        'maintenance_percent': 0.002,
        'vacancy_rate': 0.125,
        # 'rent_appreciation_yoy': 0.0,
        'one_time_other': 20000.00,

        # Growth & Taxes
        'holding_years': 5,             # Global default holding period
        'appreciation_yoy': 0.033,      # 3%
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
            # {
            #     'name': 'Rama', 
            #     'mortgage_type': 'amortizing', 
            #     'down_percent': 0.20,
            #     'base_interest_rate': 0.04,
            #     'loan_duration_years': 30,
            #     'advance_payment': 5000.00,
            #     'one_time_other': 20000.00,
            #     'closing_credits': 0.00,
            #     'points_purchased': 0,
            #     'pmi_monthly': 0.00
            # },
            {
                'name': 'SBLOC_paydown', 
                'mortgage_type': 'interest_only', 
                'down_percent': 0.0, 
                'base_interest_rate': 0.049,
                'advance_payment': 5000.00,
                # 'one_time_other': 20000.00,   
                'closing_credits': 0.00,
                'interest_only_period_years': 5,  # Configurable I/O period
                'positive_cashflow_strategy': 'pay_down_loan'
            },
            # {
            #     'name': 'SBLOC_paydown_20%', 
            #     'mortgage_type': 'interest_only', 
            #     'down_percent': 0.20, 
            #     'base_interest_rate': 0.0475,
            #     'advance_payment': 5000.00,
            #     'one_time_other': 20000.00,
            #     'closing_credits': 0.00,
            #     'interest_only_period_years': 3,  # Configurable I/O period
            #     'positive_cashflow_strategy': 'pay_down_loan'
            # },
            {
                'name': 'Morg', 
                'mortgage_type': 'amortizing', 
                'down_percent': 0.0,
                'base_interest_rate': 0.058,
                'loan_duration_years': 30,
                'advance_payment': 5000.00,
                # 'one_time_other': 20000.00,
                'closing_credits': 0.00,
                'points_purchased': 0,
                'pmi_monthly': 0.00
            },
            {
                'name': 'Morg_20', 
                'mortgage_type': 'amortizing', 
                'down_percent': 0.20,
                'base_interest_rate': 0.058,
                'loan_duration_years': 30,
                'advance_payment': 5000.00,
                # 'one_time_other': 20000.00,
                'closing_credits': 0.00,
                'points_purchased': 0,
                'pmi_monthly': 0.00
            },
            {
                'name': 'Morg_25', 
                'mortgage_type': 'amortizing', 
                'down_percent': 0.25,
                'base_interest_rate': 0.058,
                'loan_duration_years': 30,
                'advance_payment': 5000.00,
                # 'one_time_other': 20000.00,
                'closing_credits': 0.00,
                'points_purchased': 0,
                'pmi_monthly': 0.00
            },
            # {
            #     'name': 'SBLOC_1031', 
            #     'mortgage_type': 'interest_only', 
            #     'down_percent': 0.0, 
            #     'base_interest_rate': 0.0475,
            #     'advance_payment': 5000.00,
            #     'one_time_other': 20000.00,
            #     'closing_credits': 0.00,
            #     'interest_only_period_years': 3,
            #     'positive_cashflow_strategy': 'pay_down_loan',
            #     'exit_strategy': '1031_exchange'  # Defer taxes via 1031
            # },
                       
        ]
    }

    # -------------------------------------------------------------------------
    # DEAL 2: DRhorton Unit
    # -------------------------------------------------------------------------
    deal2 = {
        'deal_name': 'DR_horton_unit',
        
        # Property Details
        'sqft': 2035,
        'quote_price': 374000.00,
        'fair_market_value': 390000.00,
        'rent_per_sqft': 1.0,
        
        # Expenses & Reserves
        'prop_tax_rate': 0.009,
        'hoa_monthly': 29.167,           # Lower HOA
        'home_ins_monthly': 125.00,
        'prop_management_percent': 0.07,
        'safety_deposit_months': 2,
        'capex_percent': 0.00,
        'maintenance_percent': 0.002,
        'one_time_other': 20000.00,
        
        # Growth & Taxes
        'holding_years': 5,
        'appreciation_yoy': 0.033,
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
                'base_interest_rate': 0.049,
                'advance_payment': 5000.00,
                # 'one_time_other': 20000.00,       
                'closing_credits': 0.00,
                'interest_only_period_years': 3  # Configurable I/O period
            },
            {
                'name': 'Morg', 
                'mortgage_type': 'amortizing', 
                'down_percent': 0.0,
                'base_interest_rate': 0.058,
                'loan_duration_years': 30,
                'advance_payment': 5000.00,
                # 'one_time_other': 20000.00,       
                'closing_credits': 0.00,
                'points_purchased': 0,
                'pmi_monthly': 0.00
            },
            {
                'name': 'Morg_20', 
                'mortgage_type': 'amortizing', 
                'down_percent': 0.20,
                'base_interest_rate': 0.058,
                'loan_duration_years': 30,
                'advance_payment': 5000.00,
                # 'one_time_other': 20000.00,       
                'closing_credits': 0.00,
                'points_purchased': 0,
                'pmi_monthly': 0.00
            },
            {
                'name': 'Morg_25', 
                'mortgage_type': 'amortizing', 
                'down_percent': 0.25,
                'base_interest_rate': 0.058,
                'loan_duration_years': 30,
                'advance_payment': 5000.00,
                # 'one_time_other': 20000.00,       
                'closing_credits': 0.00,
                'points_purchased': 0,
                'pmi_monthly': 0.00
            },
            # {
            #     'name': 'SBLOC 5Y_20%', 
            #     'mortgage_type': 'interest_only', 
            #     'down_percent': 0.20, 
            #     'base_interest_rate': 0.045,
            #     'advance_payment': 5000.00,
            #     'one_time_other': 20000.00,
            #     'closing_credits': 0.00,
            #     'interest_only_period_years': 3  # Configurable I/O period
            # },
            # {
            #     'name': 'Morg_30Y_1', 
            #     'mortgage_type': 'amortizing', 
            #     'down_percent': 0.0,
            #     'base_interest_rate': 0.058,
            #     'loan_duration_years': 30,
            #     'advance_payment': 5000.00,
            #     'one_time_other': 20000.00,
            #     'closing_credits': 0.00,
            #     'points_purchased': 0,
            #     'pmi_monthly': 0.00
            # }
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
    properties = [deal1,deal2]
    
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