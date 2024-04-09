import os
from main import create_chain, create_agent
from langsmith import Client
import uuid 
from langchain.smith import RunEvalConfig

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com" # Update with your API URL if using a hosted instance of Langsmith.
os.environ["LANGCHAIN_API_KEY"] = "ls__7896a9c5ebc64582bc391d854fe1d83d" # Update with your API key

examples = [
    ("What's the minimum number of shareholders needed to start a private company in Nigeria?",
     '''The minimum number of shareholders needed to start a private company in Nigeria is generally one (1). Key Points to note:
     Single-Shareholder Companies: Under the Companies and Allied Matters Act 2020, Section 18(2)  allows a single person to form a private company, being the sole shareholder, director, and promoter.Small Companies: This provision usually applies to ‘small companies’. As provided in Section 394(3) of the CAMA,  a small company is one that meets the following criteria:
     Private: It's a private company (not publicly traded).
     Turnover and Assets: It has a maximum annual turnover of N120 million and net assets of up to N60 million (or amounts adjusted by the Corporate Affairs Commission).
     Ownership: None of the members are foreigners, governments, or government-related agencies.
     Share Control: In companies with share capital, the directors hold at least 51% of the equity shares.
     Exception for Foreign Companies: If a foreign company wants to establish a private subsidiary in Nigeria, it still needs at least two directors and a shareholder.
     Public Companies: Public companies require three (3) independent directors under Section 275(1).'''),
    ("Does a foreign national need a Nigerian partner to open a business?",
    '''Mostly Yes. Foreign nationals can establish and fully own businesses in most sectors of the Nigerian economy.  There is no strict requirement for a Nigerian partner.
    Section 78 mandates the registration of foreign companies in Nigeria and Section 80 outlines specific exceptions to this requirement.
    Section 17 of the NIPC Act further permits foreigners to own and operate businesses in most sectors. The NIPC however maintains a "negative list" detailing sectors where foreign ownership may be limited or prohibited (such as arms and ammunition production, production of narcotic drugs).
    While not mandatory, partnering with a Nigerian can offer benefits like local expertise in terms of understanding local regulations, market knowledge, and business practices and Government Relations in terms of building relationships with relevant government agencies and obtaining permits'''),
    ("What are Annual returns",
    '''In Nigeria, Annual returns in the context of the Corporate Affairs Commission are a yearly statement that all registered companies in Nigeria must submit to the Corporate Affairs Commission (CAC). They ensure transparency, update CAC records, and contribute to regulatory oversight of companies. 
    This filing provides updated information about the company's:
    Directors and shareholders
    Shareholdings and transfers
    Registered address
    Financial position (audited accounts for larger companies)
    Primarily, Section 417 to 424 of the Act of the Companies and Allied Matters Act (CAMA) 2020 outline the requirements and procedures for filing annual returns.
    As provided in Section 421, Companies generally have to file their annual returns within 42 days of their Annual General Meeting (AGM), and Newly registered companies have an 18 month grace period for their first return. Small companies are however eligible for simplified annual return filings.
    Late or non-filing of annual returns attracts penalties as seen in Section 425(1) and can even lead to a company being struck off the register on Swction 425(3)'''),
    ("How frequently do I need to update my company information with the CAC?",
    '''You need to update your company information with the Corporate Affairs Commission (CAC) in Nigeria under the following circumstances:
    Changes to Key Information: Any major change to your company's structure must be promptly reported to the CAC. This includes:
    Changes in directors or shareholders
    Changes in shareholding structure or share transfers
    Changes to the company's registered address
    Changes to the Memorandum and Articles of Association (MemArt)
    Changes in the company's nature of business
    Annual Returns: Annual returns are a mandatory yearly filing with updated company information,  even if no significant changes have occurred. Companies must file within 42 days of their Annual General Meeting (AGM).
    CAC Requests:  The CAC may periodically request that companies update specific information as part of regulatory compliance checks.''')
]

client = Client()

dataset_name = f"Legal Queries {str(uuid.uuid4())}"
dataset = client.create_dataset(dataset_name=dataset_name)
for q, a in examples:
    client.create_example(inputs={"input": q}, outputs={"answer": a}, dataset_id=dataset.id)

eval_config = RunEvalConfig(
    # We will use the chain-of-thought Q&A correctness evaluator
    evaluators=["cot_qa"],
)

results = client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=create_agent(),
    evaluation=eval_config
)
project_name = results["project_name"]