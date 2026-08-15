import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"

V5_TRAIN = DATASET / "brainrot_v5_train.jsonl"
V5_VALIDATION = DATASET / "brainrot_v5_validation.jsonl"

V6_TRAIN = DATASET / "brainrot_v6_train.jsonl"
V6_VALIDATION = DATASET / "brainrot_v6_validation.jsonl"


def example(user, assistant):
    return {
        "messages": [
            {
                "role": "user",
                "content": user
            },
            {
                "role": "assistant",
                "content": assistant
            }
        ]
    }


# ============================================================
# V6 CORRECTION DATA
# ============================================================

corrections = [

    example(
        "What is an object in programming?",
        "An object is a bundle of data and behavior. It has properties that store data and methods that define what it can do. Basically, a software entity with stats and abilities."
    ),

    example(
        "What is an object in Python?",
        "A Python object is a value that has a type, data, and behavior. Numbers, strings, lists, functions, and class instances are all objects. Python basically treats everything like it has a type card attached."
    ),

    example(
        "What is OOP?",
        "OOP means Object-Oriented Programming. You organize code around objects that contain data and behavior. The main ideas are encapsulation, inheritance, polymorphism, and abstraction."
    ),

    example(
        "What is a class?",
        "A class is a blueprint for creating objects. It defines the data and behavior those objects can have. The class is the recipe; the object is the thing you cooked."
    ),

    example(
        "What is an API?",
        "An API is a defined interface that lets software systems communicate. One program sends a request using the API's rules, and another program returns a response. Basically, software talking to software."
    ),

    example(
        "What does REST API mean?",
        "A REST API is an HTTP-based API that commonly represents resources with URLs and uses methods such as GET, POST, PUT, PATCH, and DELETE. It's basically a standardized way for apps to exchange data over the web."
    ),

    example(
        "What is JSON?",
        "JSON is a lightweight text format for representing structured data. It uses objects, arrays, strings, numbers, booleans, and null. Basically, data wearing a very readable outfit."
    ),

    example(
        "What is MongoDB?",
        "MongoDB is a NoSQL document database. It stores data as BSON documents, which are similar to JSON objects, and groups those documents into collections."
    ),

    example(
        "What is Git?",
        "Git is a distributed version control system. It tracks changes to files so you can create commits, branches, compare versions, and recover from mistakes without sacrificing your entire project to the code gods."
    ),

    example(
        "What is GitHub?",
        "GitHub is a platform for hosting Git repositories and collaborating on software. It provides repositories, branches, pull requests, issues, Actions, and other development tools."
    ),

    example(
        "What is a Git commit?",
        "A Git commit is a saved snapshot of staged changes in a repository. It gives you a point in history that you can inspect or return to later."
    ),

    example(
        "What is a Git branch?",
        "A Git branch is a movable pointer to a line of commits. It lets you work on a feature or experiment separately from the main branch."
    ),

    example(
        "What is Python?",
        "Python is a high-level general-purpose programming language known for readable syntax and a huge ecosystem. It's used for web development, automation, data science, machine learning, scripting, and more."
    ),

    example(
        "What is recursion?",
        "Recursion is when a function calls itself to solve a smaller version of a problem. A correct recursive solution needs a base case so the function eventually stops."
    ),

    example(
        "What is JavaScript?",
        "JavaScript is a programming language widely used to make web pages interactive. It also runs outside browsers through environments such as Node.js."
    ),

    example(
        "What is React?",
        "React is a JavaScript library for building user interfaces from reusable components. It helps update the UI when application state changes."
    ),

    example(
        "What is Node.js?",
        "Node.js is a JavaScript runtime built on the V8 engine. It lets JavaScript run outside the browser and is commonly used for servers, APIs, and backend applications."
    ),

    example(
        "What is Express.js?",
        "Express.js is a lightweight web framework for Node.js. It provides routing, middleware, and request-response handling for building web servers and APIs."
    ),

    example(
        "What is JWT?",
        "JWT stands for JSON Web Token. It's a compact token format commonly used to carry claims between parties, often for authentication and authorization. A JWT is signed, not automatically encrypted."
    ),

    example(
        "What is CUDA?",
        "CUDA is NVIDIA's platform and programming model for general-purpose computing on NVIDIA GPUs. Machine-learning frameworks can use CUDA to run tensor operations on the GPU."
    ),

    example(
        "What is PyTorch?",
        "PyTorch is a machine-learning framework used for tensor computation, automatic differentiation, and training neural networks. It's heavily used in deep learning."
    ),

    example(
        "What is a tokenizer?",
        "A tokenizer converts text into tokens that a language model can process. Depending on the tokenizer, a token can represent a word piece or another chunk of text."
    ),

    example(
        "What is an LLM?",
        "LLM means Large Language Model. It's a neural network trained on large amounts of text to model language and generate text. Your tiny Qwen model is still an LLM; it just has fewer parameters than the giant monsters."
    ),

    example(
        "What is fine-tuning?",
        "Fine-tuning means continuing training on a pretrained model using a task-specific dataset. It adapts the model's behavior without training a language model from zero."
    ),

    example(
        "What is LoRA?",
        "LoRA stands for Low-Rank Adaptation. Instead of updating all the original model weights, you train small low-rank adapter matrices. It's basically a lightweight upgrade pack for the model's brain."
    ),

    example(
        "What is QLoRA?",
        "QLoRA combines a quantized base model with LoRA adapters. The base model uses lower precision to reduce memory usage while the small LoRA adapters are trained."
    ),

    example(
        "What is PEFT?",
        "PEFT means Parameter-Efficient Fine-Tuning. It adapts a pretrained model by training only a small number of additional parameters instead of updating the entire model."
    ),

    example(
        "Why do we use LoRA instead of full fine-tuning?",
        "LoRA is much more memory-efficient because only small adapter parameters are trained. The large pretrained model stays mostly frozen, making fine-tuning practical on smaller GPUs."
    ),

    example(
        "What is overfitting?",
        "Overfitting happens when a model learns the training examples too closely and performs worse on unseen data. Basically, it memorized the textbook but still panics when the teacher changes the question."
    ),

    example(
        "What is validation loss?",
        "Validation loss measures the model's performance on data that wasn't directly used to update its parameters. It's useful for checking generalization and detecting overfitting."
    ),

    example(
        "What is an epoch?",
        "One epoch is one complete pass through the training dataset. If your configuration says num_train_epochs=3, the model goes through the training dataset three times."
    ),

    example(
        "What is inference?",
        "Inference is the process of using a trained model to produce predictions or generated output. Training changes the model; inference uses what the model already learned."
    ),

    example(
        "What is temperature in text generation?",
        "Temperature controls randomness during token selection. Higher temperature usually produces more varied output, while lower temperature makes generation more predictable."
    ),

    example(
        "What is top-p sampling?",
        "Top-p sampling limits candidate tokens to the smallest group whose combined probability reaches a chosen threshold. It gives generation controlled randomness."
    ),

    example(
        "What is JSONL?",
        "JSONL means JSON Lines. Each line contains a separate valid JSON object, making it convenient for datasets because records can be processed one line at a time."
    ),

    example(
        "What is a virtual environment?",
        "A Python virtual environment creates an isolated environment for project dependencies. It prevents one project's package versions from breaking another project's setup."
    ),

    example(
        "Why should .venv be in .gitignore?",
        "A virtual environment contains installed packages and machine-specific files, so it normally should not be committed to Git. Commit your dependency specification instead."
    ),

    example(
        "What is an API in one sentence?",
        "An API is a defined interface that lets one piece of software communicate with another piece of software."
    ),

    example(
        "Why is my 0.6B model giving wrong answers?",
        "Because 0.6B parameters is a very small model compared with modern large models, and your fine-tuning dataset is also relatively small. Fine-tuning teaches patterns and behavior, but it doesn't magically turn a tiny model into an encyclopedia."
    ),

    example(
        "How do I debug a Python error?",
        "Read the traceback from the bottom up, find the exception type and the line that caused it, then inspect the values involved. Don't randomly delete code and call it debugging. That's archaeology."
    ),
]


# ============================================================
# LOAD V5
# ============================================================

with open(V5_TRAIN, "r", encoding="utf-8") as f:
    train_data = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

with open(V5_VALIDATION, "r", encoding="utf-8") as f:
    validation_data = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


# ============================================================
# ADD V6 CORRECTIONS
# ============================================================

train_data.extend(corrections)


# ============================================================
# SAVE V6
# ============================================================

with open(V6_TRAIN, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(
            json.dumps(item, ensure_ascii=False) + "\n"
        )


with open(V6_VALIDATION, "w", encoding="utf-8") as f:
    for item in validation_data:
        f.write(
            json.dumps(item, ensure_ascii=False) + "\n"
        )


print("=" * 60)
print("BRAINROT V6 DATASET GENERATED")
print("=" * 60)

print("V5 training examples:", len(train_data) - len(corrections))
print("V6 correction examples:", len(corrections))
print("Final training examples:", len(train_data))
print("Validation examples:", len(validation_data))

print()
print("Saved:")
print(V6_TRAIN)
print(V6_VALIDATION)

print("=" * 60)