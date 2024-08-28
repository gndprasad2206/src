import helpers.classifier_helper as ch

def main():
    # message = input("Enter a message to classify: ")
    message = "... this is a resume ... supposed to receive not enough information ..."
    result, probs = ch.classify_message(message)

    print(f"The message is classified as: {result}")
    print("Class probabilities:")
    for index, prob in enumerate(probs):
        print(f"{ch.class_labels[index]}: {prob:.4f}")

if __name__ == "__main__":
    main()

