import math
import json

class SolicitorsRemunerationCalculator:
    """律师费报酬计算器 (SRO 2023)"""
    
    @staticmethod
    def calculate_table_a(property_value: float) -> float:
        """First Schedule - Table A (Standard Scale Fees)"""
        if property_value <= 0:
            return 0.0
            
        fee = 0.0
        # 第一阶梯: 首 500,000 (1.25%)
        tier_1_value = min(property_value, 500000)
        fee += tier_1_value * 0.0125
        
        # 第二阶梯: 接下来 7,000,000 (500,000.01 - 7,500,000) (1.00%)
        if property_value > 500000:
            tier_2_value = min(property_value - 500000, 7000000)
            fee += tier_2_value * 0.01
            
        # 第三阶梯: 超过 7,500,000 (最高 1.00%，可协商，这里按最高值计算以做预算)
        if property_value > 7500000:
            tier_3_value = property_value - 7500000
            fee += tier_3_value * 0.01 
            print(f"注意: 超过 RM7,500,000 的部分律师费可协商，此处按最高 1% 计算。")
            
        return max(fee, 500.0) # 最低收费 RM500

    @staticmethod
    def calculate_table_b(property_value: float) -> float:
        """First Schedule - Table B (HDA Transactions)"""
        if property_value <= 0:
            return 0.0
            
        # 根据档位确定 Table A 的折扣比例
        if property_value <= 50000:
            return 500.0 # 固定收费
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
        
        return max(discounted_fee, 500.0) # 最低收费 RM500


class TenancyStampDutyCalculator:
    """租约印花税计算器 (基于 2026 年新规)"""
    
    BASE_UNIT = 250
    FLAT_FEE_EXTRA_COPY = 10
    
    @staticmethod
    def calculate(monthly_rental: float, duration_years: float, extra_copies: int = 0) -> float:
        annual_rental = monthly_rental * 12
        # 2026年起取消 RM2400 豁免额度
        taxable_rental = annual_rental 
        
        # 向上取整至最近的 250
        calculation_units = math.ceil(taxable_rental / TenancyStampDutyCalculator.BASE_UNIT)
        
        # 决定费率倍数
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
    """租约管理费计算器"""
    
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
    """员工解雇福利计算器"""
    
    @staticmethod
    def calculate(monthly_salary: float, service_years: float) -> float:
        # 计算日薪 (全年薪水 / 365)
        daily_wage = (monthly_salary * 12) / 365
        
        # 根据服务年限决定每年补偿天数
        if service_years < 2:
            days_per_year = 10
        elif service_years < 5:
            days_per_year = 15
        else:
            days_per_year = 20
            
        # 计算总福利金额
        total_benefit = daily_wage * service_years * days_per_year
        return round(total_benefit, 2)


# ==========================================
# 测试与使用示例 (Usage Examples)
# ==========================================
if __name__ == "__main__":
    print("--- 1. 律师费计算测试 (SRO 2023) ---")
    property_val = 800000
    print(f"RM {property_val:,.2f} 房产的标准律师费 (Table A): RM {SolicitorsRemunerationCalculator.calculate_table_a(property_val):,.2f}")
    print(f"RM {property_val:,.2f} 房产的发展商项目律师费 (Table B): RM {SolicitorsRemunerationCalculator.calculate_table_b(property_val):,.2f}")
    print()
    
    print("--- 2. 租约印花税测试 (2026新规) ---")
    rent = 2500
    duration = 2.5
    copies = 2
    duty = TenancyStampDutyCalculator.calculate(rent, duration, copies)
    print(f"月租 RM {rent}, 租期 {duration} 年, {copies} 份副本 -> 印花税总额: RM {duty:,.2f}")
    print()
    
    print("--- 3. 租约管理费测试 ---")
    rent_admin = 2500
    print(f"月租 RM {rent_admin} 的管理费为: RM {TenancyAdminFeeCalculator.calculate(rent_admin):,.2f}")
    print()
    
    print("--- 4. 员工解雇福利测试 ---")
    salary = 5000
    years = 3.5
    benefit = EmploymentTerminationCalculator.calculate(salary, years)
    print(f"月薪 RM {salary}, 服务 {years} 年 -> 解雇福利赔偿: RM {benefit:,.2f}")