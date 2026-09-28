"""Create a small SYNTHETIC dataset so the pipeline runs end-to-end.

For real results download the EMSCAD dataset (fake_job_postings.csv) from Kaggle
("Real / Fake Job Posting Prediction") and place it in data/. Same column names.
"""
import random
import pandas as pd

random.seed(42)
REAL_TITLES = ["Software Engineer", "Data Analyst", "Marketing Manager", "Accountant", "Nurse", "Sales Executive", "HR Coordinator", "Mechanical Engineer"]
FAKE_TITLES = ["Work From Home Data Entry", "Earn Money Online Typist", "Home Based Payment Processor", "Urgent Hiring No Experience", "Personal Assistant Wire Transfer"]
REAL_DESC = ["Design, develop and test software components with the engineering team.", "Analyse business data and prepare monthly reports for management.", "Plan campaigns, manage budgets and coordinate with agencies.", "Maintain ledgers, reconcile accounts and support audits.", "Provide patient care and maintain accurate clinical records."]
REAL_REQ = ["Bachelor degree in a relevant field and 2 years experience.", "Strong communication skills and proficiency with office tools.", "Experience with SQL, Python or similar tools.", "Professional certification preferred."]
REAL_BEN = ["Health insurance, paid leave and annual bonus.", "Provident fund and learning allowance.", "Flexible hours and career growth."]
FAKE_DESC = ["Earn 5000 dollars weekly from home no experience needed. Send your bank details to start immediately.", "Urgent hiring! Pay registration fee to receive your training kit. Contact us on whatsapp gmail.com today.", "Process payments from home, keep a commission, we transfer funds to your personal account.", "Limited slots, apply now, no interview required, guaranteed income, click http://easy-cash.biz to register."]
FAKE_REQ = ["No qualification required. Must have bank account and internet.", "Send SSN and passport copy with application.", "Anyone can apply, immediate joining."]
FAKE_BEN = ["Unlimited earning, instant weekly payout, no targets.", "Huge commission and free laptop after deposit."]
COMPANIES = ["Acme Technologies builds enterprise software for global clients.", "BrightCare Hospitals is a multi-specialty healthcare provider.", "Northwind Traders is an established retail and logistics company."]

rows = []
for i in range(1200):
    fake = random.random() < 0.15
    if fake:
        rows.append(dict(title=random.choice(FAKE_TITLES), company_profile="" if random.random() < 0.7 else "Global Opportunity Group",
                         description=random.choice(FAKE_DESC), requirements=random.choice(FAKE_REQ), benefits=random.choice(FAKE_BEN),
                         employment_type=random.choice(["Part-time", "Other", ""]), required_experience="Not Applicable", required_education="",
                         industry="", function="Administrative", fraudulent=1))
    else:
        rows.append(dict(title=random.choice(REAL_TITLES), company_profile=random.choice(COMPANIES),
                         description=random.choice(REAL_DESC), requirements=random.choice(REAL_REQ), benefits=random.choice(REAL_BEN),
                         employment_type="Full-time", required_experience=random.choice(["Entry level", "Mid-Senior level"]),
                         required_education="Bachelor Degree", industry=random.choice(["IT", "Healthcare", "Finance", "Retail"]),
                         function=random.choice(["Engineering", "Sales", "Finance", "Health Care Provider"]), fraudulent=0))
pd.DataFrame(rows).to_csv("data/sample_job_postings.csv", index=False)
print("Wrote data/sample_job_postings.csv", len(rows), "rows")
