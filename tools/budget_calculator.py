from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ExpenseCategory(Enum):
    ATTRACTION = "attraction"
    TRANSPORT = "transport"
    FOOD = "food"
    ACCOMMODATION = "accommodation"
    SHOPPING = "shopping"
    OTHER = "other"

@dataclass
class Expense:
    """Individual expense item"""
    name: str
    category: ExpenseCategory
    cost: float
    currency: str = "USD"
    notes: str = ""
    day: Optional[int] = None

@dataclass
class BudgetSummary:
    """Budget summary with breakdown"""
    total_cost: float
    expenses: List[Expense]
    category_breakdown: Dict[str, float]
    daily_breakdown: Dict[int, float]
    recommendations: List[str]
    margin_amount: float
    final_budget: float

class BudgetCalculator:
    """Budget calculation and management tool"""
    
    def __init__(self, margin_percentage: float = 0.15):
        self.margin_percentage = margin_percentage
        self.expenses: List[Expense] = []
        
        # Default cost estimates (USD)
        self.default_costs = {
            "local_transport_day": 15,
            "taxi_per_km": 1.5,
            "budget_meal": 12,
            "mid_range_meal": 25,
            "fine_dining": 60,
            "coffee": 4,
            "water_bottle": 2,
            "souvenir": 15,
            "guidebook": 20
        }
    
    def add_expense(self, expense: Expense) -> None:
        """Add an expense to the budget"""
        self.expenses.append(expense)
        logger.info(f"Added expense: {expense.name} - ${expense.cost}")
    
    def add_attraction_costs(self, attractions: List[Dict]) -> None:
        """Add attraction costs from PlaceFinder results"""
        for i, attraction in enumerate(attractions):
            expense = Expense(
                name=attraction['name'],
                category=ExpenseCategory.ATTRACTION,
                cost=attraction.get('avg_cost_usd', 0),
                notes=f"Category: {attraction.get('category', 'Unknown')}"
            )
            self.add_expense(expense)
    
    def add_transport_cost(self, transport_type: str, amount: float, day: Optional[int] = None) -> None:
        """Add transport costs"""
        cost_map = {
            "local_transport": self.default_costs["local_transport_day"],
            "taxi": amount * self.default_costs["taxi_per_km"],
            "public_transport": self.default_costs["local_transport_day"],
            "walking": 0
        }
        
        cost = cost_map.get(transport_type, amount)
        expense = Expense(
            name=f"Transport - {transport_type.title()}",
            category=ExpenseCategory.TRANSPORT,
            cost=cost,
            day=day,
            notes=f"Type: {transport_type}"
        )
        self.add_expense(expense)
    
    def add_meal_costs(self, meal_type: str, count: int = 1, day: Optional[int] = None) -> None:
        """Add meal costs based on type"""
        cost_map = {
            "budget": self.default_costs["budget_meal"],
            "mid_range": self.default_costs["mid_range_meal"],
            "fine_dining": self.default_costs["fine_dining"],
            "street_food": self.default_costs["budget_meal"] * 0.6,
            "fast_food": self.default_costs["budget_meal"] * 0.8
        }
        
        unit_cost = cost_map.get(meal_type, self.default_costs["mid_range_meal"])
        total_cost = unit_cost * count
        
        expense = Expense(
            name=f"Meals - {meal_type.title()} ({count}x)",
            category=ExpenseCategory.FOOD,
            cost=total_cost,
            day=day,
            notes=f"Type: {meal_type}, Quantity: {count}"
        )
        self.add_expense(expense)
    
    def estimate_daily_costs(self, activity_count: int, meal_preference: str = "mid_range") -> float:
        """Estimate basic daily costs"""
        daily_cost = 0
        
        # Transport (assuming local transport for the day)
        daily_cost += self.default_costs["local_transport_day"]
        
        # Meals (breakfast, lunch, dinner)
        meal_cost = {
            "budget": self.default_costs["budget_meal"],
            "mid_range": self.default_costs["mid_range_meal"],
            "luxury": self.default_costs["fine_dining"]
        }.get(meal_preference, self.default_costs["mid_range_meal"])
        daily_cost += meal_cost * 3
        
        # Drinks and snacks
        daily_cost += self.default_costs["coffee"] * 2
        daily_cost += self.default_costs["water_bottle"]
        
        # Miscellaneous (souvenirs, tips, etc.)
        daily_cost += self.default_costs["souvenir"] * 0.5
        
        return daily_cost
    
    def calculate_summary(self) -> BudgetSummary:
        """Calculate comprehensive budget summary"""
        if not self.expenses:
            return BudgetSummary(
                total_cost=0,
                expenses=[],
                category_breakdown={},
                daily_breakdown={},
                recommendations=["No expenses recorded yet"],
                margin_amount=0,
                final_budget=0
            )
        
        total_cost = sum(expense.cost for expense in self.expenses)
        margin_amount = total_cost * self.margin_percentage
        final_budget = total_cost + margin_amount
        
        # Category breakdown
        category_breakdown = {}
        for category in ExpenseCategory:
            category_total = sum(
                expense.cost for expense in self.expenses 
                if expense.category == category
            )
            if category_total > 0:
                category_breakdown[category.value] = category_total
        
        # Daily breakdown
        daily_breakdown = {}
        for expense in self.expenses:
            if expense.day is not None:
                if expense.day not in daily_breakdown:
                    daily_breakdown[expense.day] = 0
                daily_breakdown[expense.day] += expense.cost
        
        # Generate recommendations
        recommendations = self._generate_recommendations(total_cost, category_breakdown)
        
        summary = BudgetSummary(
            total_cost=total_cost,
            expenses=self.expenses.copy(),
            category_breakdown=category_breakdown,
            daily_breakdown=daily_breakdown,
            recommendations=recommendations,
            margin_amount=margin_amount,
            final_budget=final_budget
        )
        return asdict(summary)
    
    def _generate_recommendations(self, total_cost: float, category_breakdown: Dict[str, float]) -> List[str]:
        """Generate budget recommendations"""
        recommendations = []
        
        if total_cost == 0:
            return ["Add expenses to get budget recommendations"]
        
        # Check category distribution
        if category_breakdown.get("food", 0) > total_cost * 0.4:
            recommendations.append("Food costs are high - consider local eateries or street food")
        
        if category_breakdown.get("transport", 0) > total_cost * 0.3:
            recommendations.append("Transport costs are significant - look for day passes or walking routes")
        
        if category_breakdown.get("attraction", 0) < total_cost * 0.2:
            recommendations.append("Consider adding more attractions to make the most of your trip")
        
        # Budget level recommendations
        daily_average = total_cost / max(len(set(e.day for e in self.expenses if e.day)), 1)
        
        if daily_average < 50:
            recommendations.append("Budget-friendly trip - great for backpackers")
        elif daily_average > 150:
            recommendations.append("Luxury travel budget - consider premium experiences")
        else:
            recommendations.append("Mid-range budget - good balance of comfort and value")
        
        # General tips
        recommendations.extend([
            f"Added {self.margin_percentage:.0%} buffer for unexpected expenses",
            "Keep receipts and track spending during your trip",
            "Consider travel insurance for peace of mind"
        ])
        
        return recommendations
    
    def optimize_budget(self, max_budget: float) -> Dict:
        """Suggest budget optimizations to meet target budget"""
        current_total = sum(expense.cost for expense in self.expenses)
        final_budget = current_total * (1 + self.margin_percentage)
        
        if final_budget <= max_budget:
            return {
                "status": "within_budget",
                "message": f"Your budget of ${final_budget:.2f} is within the limit of ${max_budget:.2f}",
                "savings": max_budget - final_budget
            }
        
        overage = final_budget - max_budget
        suggestions = []
        
        # Suggest specific cuts
        category_breakdown = {}
        for category in ExpenseCategory:
            total = sum(e.cost for e in self.expenses if e.category == category)
            if total > 0:
                category_breakdown[category.value] = total
        
        if category_breakdown.get("food", 0) > current_total * 0.3:
            potential_savings = category_breakdown["food"] * 0.3
            suggestions.append(f"Reduce dining costs by ${potential_savings:.2f} (choose local/street food)")
        
        if category_breakdown.get("transport", 0) > current_total * 0.25:
            potential_savings = category_breakdown["transport"] * 0.2
            suggestions.append(f"Save ${potential_savings:.2f} on transport (use public transport/walk more)")
        
        return {
            "status": "over_budget",
            "overage": overage,
            "suggestions": suggestions,
            "current_total": final_budget,
            "target": max_budget
        }
    
    def reset(self) -> None:
        """Reset all expenses"""
        self.expenses.clear()
        logger.info("Budget calculator reset")
    
    def get_expense_by_category(self, category: ExpenseCategory) -> List[Expense]:
        """Get all expenses for a specific category"""
        return [e for e in self.expenses if e.category == category]
    
    def to_dict(self) -> Dict:
        """Convert budget data to dictionary"""
        summary = self.calculate_summary()
        return summary