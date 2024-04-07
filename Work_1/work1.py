import re

while True:
    expression_enter = input("Hello! Enter expression, like that: 12 + 53 \n")

    numbers = re.findall(r'-?\d+', expression_enter)
    if not numbers or len(numbers) < 2:
        print("Error! Enter valid data with at least two numbers!")
        continue 

    numbers = [int(number) for number in numbers]

    symbols = re.findall(r'[\+\-\*/]', expression_enter)
    if not symbols:
        print("Error! Enter valid data with a mathematical symbol!")
        continue 

    if symbols[0] == '+':
        ans = numbers[0] + numbers[1]
    elif symbols[0] == '-':
        ans = numbers[0] - numbers[1]
    elif symbols[0] == '/':
        if numbers[1] == 0:
            print("Error! Division by zero is not allowed!")
            continue
        ans = numbers[0] / numbers[1]
    elif symbols[0] == '*':
        ans = numbers[0] * numbers[1]

    print("The answer is equal to", ans)
    break 