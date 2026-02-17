"""
Seed Social Proof Engine with realistic demo data.
Run: python -m scripts.seed_social_proof
"""

import sys
import time

sys.path.insert(0, ".")

from src.db.session_db import session_db

# Company stats (top Indian IT companies + MNCs)
companies = [
    {"company_name": "TCS", "enrollments_count": 45, "last_enrollment_date": "2025-02-10", "notable_outcomes": "3 got promoted to AI leads within 4 months"},
    {"company_name": "Wipro", "enrollments_count": 32, "last_enrollment_date": "2025-02-14", "notable_outcomes": "One team now uses AI for all client proposals"},
    {"company_name": "Infosys", "enrollments_count": 28, "last_enrollment_date": "2025-02-12", "notable_outcomes": "2 transitioned to AI/ML teams internally"},
    {"company_name": "Cognizant", "enrollments_count": 22, "last_enrollment_date": "2025-02-08", "notable_outcomes": ""},
    {"company_name": "Accenture", "enrollments_count": 19, "last_enrollment_date": "2025-02-11", "notable_outcomes": "One member automated their reporting workflow"},
    {"company_name": "HCL", "enrollments_count": 15, "last_enrollment_date": "2025-01-28", "notable_outcomes": ""},
    {"company_name": "Tech Mahindra", "enrollments_count": 12, "last_enrollment_date": "2025-02-05", "notable_outcomes": ""},
    {"company_name": "Deloitte", "enrollments_count": 11, "last_enrollment_date": "2025-01-30", "notable_outcomes": ""},
    {"company_name": "Amazon", "enrollments_count": 14, "last_enrollment_date": "2025-02-13", "notable_outcomes": ""},
    {"company_name": "Google", "enrollments_count": 8, "last_enrollment_date": "2025-02-09", "notable_outcomes": ""},
    {"company_name": "Microsoft", "enrollments_count": 7, "last_enrollment_date": "2025-02-01", "notable_outcomes": ""},
    {"company_name": "Flipkart", "enrollments_count": 6, "last_enrollment_date": "2025-01-25", "notable_outcomes": ""},
    {"company_name": "Capgemini", "enrollments_count": 10, "last_enrollment_date": "2025-02-06", "notable_outcomes": ""},
    {"company_name": "Zoho", "enrollments_count": 5, "last_enrollment_date": "2025-02-04", "notable_outcomes": ""},
    {"company_name": "Freshworks", "enrollments_count": 4, "last_enrollment_date": "2025-01-22", "notable_outcomes": ""},
]

# City stats (major Indian cities)
cities = [
    {"city_name": "Hyderabad", "enrollments_count": 340, "trending": 1},
    {"city_name": "Bangalore", "enrollments_count": 290, "trending": 1},
    {"city_name": "Mumbai", "enrollments_count": 250, "trending": 0},
    {"city_name": "Pune", "enrollments_count": 220, "trending": 1},
    {"city_name": "Chennai", "enrollments_count": 180, "trending": 0},
    {"city_name": "Delhi", "enrollments_count": 170, "trending": 0},
    {"city_name": "Noida", "enrollments_count": 150, "trending": 1},
    {"city_name": "Gurgaon", "enrollments_count": 130, "trending": 0},
    {"city_name": "Kolkata", "enrollments_count": 90, "trending": 0},
    {"city_name": "Ahmedabad", "enrollments_count": 70, "trending": 0},
    {"city_name": "Jaipur", "enrollments_count": 45, "trending": 0},
    {"city_name": "Kochi", "enrollments_count": 55, "trending": 1},
]

# Role stats
roles = [
    {"role_name": "Software Engineer", "enrollments_count": 180, "avg_salary_increase": "45%", "success_stories": "Someone from your exact role built an AI SaaS tool and doubled their freelance income"},
    {"role_name": "Data Analyst", "enrollments_count": 120, "avg_salary_increase": "50%", "success_stories": "A data analyst from Pune automated 70% of their reporting using what they learned"},
    {"role_name": "Product Manager", "enrollments_count": 85, "avg_salary_increase": "35%", "success_stories": "One PM now specs AI features directly, their team ships 2x faster"},
    {"role_name": "Student", "enrollments_count": 420, "avg_salary_increase": "", "success_stories": "A final-year student built their capstone with AI and got placed at Amazon"},
    {"role_name": "Fresher", "enrollments_count": 310, "avg_salary_increase": "", "success_stories": "Multiple freshers landed their first job citing AI skills from this program"},
    {"role_name": "Designer", "enrollments_count": 65, "avg_salary_increase": "40%", "success_stories": "A UX designer now creates entire prototypes with AI in hours instead of days"},
    {"role_name": "Marketing Manager", "enrollments_count": 55, "avg_salary_increase": "30%", "success_stories": "One marketing manager 3x'd their content output using AI tools from the program"},
    {"role_name": "Consultant", "enrollments_count": 40, "avg_salary_increase": "55%", "success_stories": "A consultant added AI advisory services and doubled their billing rate"},
    {"role_name": "Team Lead", "enrollments_count": 35, "avg_salary_increase": "25%", "success_stories": ""},
    {"role_name": "Freelancer", "enrollments_count": 95, "avg_salary_increase": "60%", "success_stories": "Freelancers report 2-3x income increase after adding AI services"},
    {"role_name": "Entrepreneur", "enrollments_count": 75, "avg_salary_increase": "", "success_stories": "An entrepreneur automated their entire customer support with AI bots"},
    {"role_name": "Cloud Engineer", "enrollments_count": 25, "avg_salary_increase": "40%", "success_stories": "Someone from your exact role completed the program and got promoted within 2 months"},
]

print("Seeding social proof data...")

for item in companies:
    name = item.pop("company_name")
    session_db.upsert_social_proof_company(name, **item)
print(f"  {len(companies)} companies")

for item in cities:
    name = item.pop("city_name")
    session_db.upsert_social_proof_city(name, **item)
print(f"  {len(cities)} cities")

for item in roles:
    name = item.pop("role_name")
    session_db.upsert_social_proof_role(name, **item)
print(f"  {len(roles)} roles")

# Give background writer time to flush
time.sleep(2)
print("Done! Social proof data seeded.")
