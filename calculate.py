import math
import json

class SolicitorsRemunerationCalculator:
    """Solicitors' Remuneration Calculator (SRO 2023)"""
    
    @staticmethod
    def calculate_table_a(property_value: float) -> float:
        """First Schedule - Table A (Standard Scale Fees)"""
        if property_value <= 0:
            return 0.0
            
        fee = 0.0
        # Tier 1: First 500,000 (1.25%)
        tier_1_value = min(property_value, 500000)
        fee += tier_1_value * 0.0125
        
        # Tier 2: Next 7,000,000 (500,000.01 - 7,500,000) (1.00%)
        if property_value > 500000:
            tier_2_value = min(property_value - 500000, 7000000)
            fee += tier_2_value * 0.01
            
        # Tier 3: Exceeding 7,500,000 (Maximum 1.00%, negotiable, calculated at the maximum rate here for budgeting)
        if property_value > 7500000:
            tier_3_value = property_value - 7500000
            fee += tier_3_value * 0.01 
            print(f"Note: Legal fees for the portion exceeding RM7,500,000 are negotiable. Calculated at the maximum 1% here.")
            
        return max(fee, 500.0) # Minimum fee RM500

    @staticmethod
    def calculate_table_b(property_value: float) -> float:
        """First Schedule - Table B (HDA Transactions)"""
        if property_value <= 0:
            return 0.0
            
        # Determine the discount percentage for Table A based on the value tier
        if property_value <= 50000:
            return 500.0 # Flat fee
        elif property_value <= 250000:
            percentage = 0.75
        elif property_value <= 500000:
            percentage = 0.70
        elif property_value <= 1000000:
            percentage = 0.65
        else:
            percentage = 0.50
            
        table_a_fee = SolicitorsRemunerationCalculator.calculate_table_a(property_value)
        discounted_fee = table_a_fee * percentage
        
        return max(discounted_fee, 500.0) # Minimum fee RM500


class TenancyStampDutyCalculator:
    """Tenancy Stamp Duty Calculator (Based on 2026 new regulations)"""
    
    BASE_UNIT = 250
    FLAT_FEE_EXTRA_COPY = 10
    
    @staticmethod
    def calculate(monthly_rental: float, duration_years: float, extra_copies: int = 0) -> float:
        annual_rental = monthly_rental * 12
        # RM2,400 exemption limit is abolished starting from 2026
        taxable_rental = annual_rental 
        
        # Round up to the nearest 250
        calculation_units = math.ceil(taxable_rental / TenancyStampDutyCalculator.BASE_UNIT)
        
        # Determine the rate multiplier based on duration
        if duration_years <= 1:
            rate_per_unit = 1
        elif duration_years <= 3:
            rate_per_unit = 3
        elif duration_years <= 5:
            rate_per_unit = 5
        else:
            rate_per_unit = 7
            
        base_stamp_duty = calculation_units * rate_per_unit
        total_stamp_duty = base_stamp_duty + (extra_copies * TenancyStampDutyCalculator.FLAT_FEE_EXTRA_COPY)
        
        return total_stamp_duty


class TenancyAdminFeeCalculator:
    """Tenancy Administration Fee Calculator"""
    
    @staticmethod
    def calculate(monthly_rental: float) -> float:
        if monthly_rental < 1000:
            return 100.0
        elif monthly_rental <= 1999:
            return 150.0
        elif monthly_rental <= 3000:
            return 200.0
        elif monthly_rental <= 4000:
            return 250.0
        else:
            return 300.0


class EmploymentTerminationCalculator:
    """Employment Termination Benefits Calculator"""
    
    @staticmethod
    def calculate(monthly_salary: float, service_years: float) -> float:
        # Calculate daily wage (Annual salary / 365)
        daily_wage = (monthly_salary * 12) / 365
        
        # Determine the number of compensation days per year based on years of service
        if service_years < 2:
            days_per_year = 10
        elif service_years < 5:
            days_per_year = 15
        else:
            days_per_year = 20
            
        # Calculate total benefit amount
        total_benefit = daily_wage * service_years * days_per_year
        return round(total_benefit, 2)


# Testing and Usage Examples
# ==========================================
if __name__ == "__main__":
    print("--- 1. Solicitors' Remuneration Calculation Test (SRO 2023) ---")
    property_val = 800000
    print(f"Standard legal fee for RM {property_val:,.2f} property (Table A): RM {SolicitorsRemunerationCalculator.calculate_table_a(property_val):,.2f}")
    print(f"HDA transaction legal fee for RM {property_val:,.2f} property (Table B): RM {SolicitorsRemunerationCalculator.calculate_table_b(property_val):,.2f}")
    print()
    
    print("--- 2. Tenancy Stamp Duty Test (2026 New Regulations) ---")
    rent = 2500
    duration = 2.5
    copies = 2
    duty = TenancyStampDutyCalculator.calculate(rent, duration, copies)
    print(f"Monthly rent RM {rent}, duration {duration} years, {copies} extra copies -> Total Stamp Duty: RM {duty:,.2f}")
    print()
    
    print("--- 3. Tenancy Administration Fee Test ---")
    rent_admin = 2500
    print(f"Administration fee for monthly rent RM {rent_admin} is: RM {TenancyAdminFeeCalculator.calculate(rent_admin):,.2f}")
    print()
    
    print("--- 4. Employment Termination Benefit Test ---")
    salary = 5000
    years = 3.5
    benefit = EmploymentTerminationCalculator.calculate(salary, years)
    print(f"Monthly salary RM {salary}, {years} years of service -> Termination Benefit Compensation: RM {benefit:,.2f}")
