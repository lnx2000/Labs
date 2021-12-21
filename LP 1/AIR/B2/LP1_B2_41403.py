from nltk.chat.util import Chat, reflections

pairs = [
            [
                r"my name is(.*)",
                ["Hello %1, how are you today?",]
            ],
            [
                r"(.*) is your name?",
                ["My name is DEE and I will help you with your financial queries today.",]
            ],
            [
                r"(.*) invest (.*)",
                ["Basically there are many options to invest- 1.Regional and 2.Stocks.\nIn which section would you like to invest?",]
            ],
            [
                r"(.*)Regional(.*)",
                ["There are many- SBI,HSBC. Which bank would you like to go for?",]
            ],
            [
                r"(.*)SBI(.*)",
                ["SBI offers 15 percent Interest.",]
            ],
            [
                r"(.*)HSBC(.*)",
                ["HSBC offers 15.5 percent Interest.",]
            ],
            [
                r"(.*)Stocks(.*)",
                ["We have 2 companies to offer: 1. APL 2. GLE.\n choose any one to know more.\n",]
            ],
            [
                r"(.*)APL(.*)",
                ["The company AAA has a ROI = 11 percent",]
            ],
            [
                r"(.*)GLE(.*)",
                ["The company BBB has a ROI = 13 percent",]
            ],
            [
                r"hi|hey|hello(.*)",
                ["Hello", "Hey there",]
            ],
            [
                r"bye",
                ["Signing out, hope to see you again!",]
            ],
            [
                r"(.*)no|thank(.*)",
                ["Okay, glad I could help. Signing out, hope to see you again!",]
            ],
            [
                r"(.*)okay(.*)",
                ["Is there anything else you would like to know? To exit, type bye",]
            ],
        ]

def chatbot():
    print("Hey I am a DEE, a ChatBot made without ML, only using NLTK library.\nPlease type in English language (lower case) what you want to ask me.\nPress Q to exit")
    chat = Chat(pairs, reflections)
    chat.converse()

if __name__ == "__main__":
    chatbot()