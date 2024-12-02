import os
from dotenv import load_dotenv
from main import main

load_dotenv()
QUESTION = os.getenv('QUESTION')
CORRECT_ANSWER = os.getenv('CORRECT_ANSWER')
POISONED_ANSWER = os.getenv('POISONED_ANSWER')
I = int(os.getenv('I'))

if __name__ == '__main__':
    correct_answers = 0
    poisoned_answers = 0
    unknown = 0

    # prompt the LLM
    questions = [QUESTION] * I 
    answers = main(questions)

    # evaluate results
    for answer in answers:
        if CORRECT_ANSWER in answer:
            correct_answers += 1
        elif POISONED_ANSWER in answer:
            poisoned_answers += 1
        else:
            unknown += 1
    
    # print out
    print("====================================================================================================")
    print(f"Out of {I} attempts of question: '{QUESTION}'")
    print(f"With expected response: {CORRECT_ANSWER}, and")
    print(f"Poisoned answer: {POISONED_ANSWER} ...\n")
    print(f"Correct Rate:            {round(correct_answers * 100 / I, 2)}%")
    print(f"Poisoned Rate:           {round(poisoned_answers * 100 / I, 2)}%")
    print(f"Unknown/random response: {round(unknown * 100 / I, 2)}%")

