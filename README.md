# LexAuto

Legal Chatbot for Lexify

## Getting Started

These instructions will help you set up and run the project on your local machine for development and testing purposes.

## Prerequisites

- Visual Studio Code
- Python
- pipenv

## Installing

1. Clone the repository to your local machine: <br>
   `git clone https://github.com/Lexauto-ai/AI_legal_Assistant.git`

2. Navigate to the project directory:<br>
   `cd your-project`
3. Open the project in Visual Studio Code: <br>
   `code .`
4. In your project directory, create .env file with your own PINECONE_API_KEY, PINECONE_ENV and OPENAI_API_KEY
5. In the terminal, create a virtual environment and install dependencies using pipenv:<br>
   `pipenv shell`<br>
   `pipenv install`
6. Run the Streamlit app: <br>
   `streamlit run frontend.py`
   <br>
   The application should now be running locally. Open your web browser and go to http://localhost:8501 to view the app.
7. Run the langsmith test:
   <br>
   `python3 test.py`
   <br>

## Built With

- Streamlit - The web framework used
- langchain

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the [License Name] - see the LICENSE.md file for details

## Acknowledgments

Hat tip to anyone whose code was used
Inspiration
