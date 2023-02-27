"""
Author: Ron Keinan
February 2023
AI course Final Project
"""
import statistics
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import train_models

def classify_email_gui():
    """
    This function is used to classify the email using the trained model.
    """
    # if no classifier is selected, show error message and exit function
    if classifier_listbox.curselection() == ():
        messagebox.showerror("Error", "Please select a classifier")
        return

    email_text = email_textbox.get("1.0", "end-1c")

    # if no text entered in the text box, show error message and exit function
    if email_text == "\n" or email_text == 'Paste email here\n':
        messagebox.showerror("Error", "Please enter an email")
        return
    if len(email_text.split()) < 4:
        messagebox.showerror("Error", "Please enter a longer email (more than 3 words)")
        return

    # get the selected classifier from the listbox
    selected_classifier = classifier_listbox.get(tk.ACTIVE)

    dataset_name = dataset_name_combobox.get()
    dataset_size = dataset_size_combobox.get()

    # get the email text from the text box
    if selected_classifier == 'AVG Ensemble':
        predictions = []
        for classifier in classifiers:
            if classifier != 'AVG Ensemble':
                prediction, preprocessed_email = train_models.classify_email(email_text, classifier, dataset_name,
                                                                             dataset_size)
                if prediction == -1:
                    # dataset not found please run train_models.py
                    messagebox.showerror("Error", "Dataset not found. Please run train_models.py")
                    return
                predictions.append(prediction)
        prediction = statistics.mode(predictions)

    else:
        prediction, preprocessed_email = train_models.classify_email(email_text, selected_classifier, dataset_name,
                                                                     dataset_size)

    if prediction == -1:
        # dataset not found please run train_models.py
        messagebox.showerror("Error", "Dataset not found. Please run train_models.py")
        return

    # write the preprocessed email to the preprocessed email text box
    preprocessed_email_textbox.config(state=tk.NORMAL)
    preprocessed_email_textbox.delete('1.0', tk.END)
    preprocessed_email_textbox.insert(tk.END, preprocessed_email)
    preprocessed_email_textbox.config(state=tk.DISABLED)

    # show the classification result
    if prediction == 1:
        messagebox.showinfo("Classification Result", "The email is spam.")
    elif prediction == 0:
        messagebox.showinfo("Classification Result", "The email is not spam.")
    else:
        messagebox.showerror("Error", "The email could not be classified.")


def update_classifier_description(event):
    """
    This function is used to update the classifier description when a classifier is selected.
    :param event: the event that triggered the function
    :return: None
    """
    # get the file of best models of the selected dataset
    file_name = dataset_name_combobox.get()
    data_size = dataset_size_combobox.get()
    best_models_file = f"trained_models/{file_name}_datasize_{data_size}/best_models.csv"
    best_models_df = pd.read_csv(best_models_file)

    # get the selected item in the listbox
    selected_classifier = classifier_listbox.get(classifier_listbox.curselection())
    # update the text box with the description and accuracy score of the selected classifier
    if selected_classifier == 'Decision Tree':
        classifier_description_textbox.config(state=tk.NORMAL)
        classifier_description_textbox.delete('1.0', tk.END)
        classifier_accuracy = str(
            best_models_df.loc[best_models_df['model'] == selected_classifier, 'accuracy'].values[0])
        classifier_description_textbox.insert(tk.END,
                                              "Decision tree is a tree-structured classifier where internal "
                                              "nodes represent the features of a dataset, "
                                              "branches represent the decision rules, "
                                              "and each leaf node represents the outcome. "
                                              "\n\nAccuracy score: " + classifier_accuracy)
        classifier_description_textbox.config(state=tk.DISABLED)
    elif selected_classifier == 'KNN':
        classifier_description_textbox.config(state=tk.NORMAL)
        classifier_description_textbox.delete('1.0', tk.END)
        classifier_accuracy = str(
            best_models_df.loc[best_models_df['model'] == selected_classifier, 'accuracy'].values[0])
        classifier_description_textbox.insert(tk.END,
                                              "The k-nearest neighbors algorithm is a non-parametric"
                                              " method used for classification and regression. "
                                              "It works by finding the k closest training examples in "
                                              "the feature space and using them to predict the class of a new sample. "
                                              "\n\nAccuracy score: " + classifier_accuracy)
        classifier_description_textbox.config(state=tk.DISABLED)
    elif selected_classifier == 'Logistic Regression':
        classifier_description_textbox.config(state=tk.NORMAL)
        classifier_description_textbox.delete('1.0', tk.END)
        classifier_accuracy = str(
            best_models_df.loc[best_models_df['model'] == selected_classifier, 'accuracy'].values[0])
        classifier_description_textbox.insert(tk.END,
                                              "Logistic regression is a statistical method for analyzing a dataset "
                                              "in which there are one or more independent variables that determine an "
                                              "outcome. The outcome is measured with a dichotomous variable. "
                                              "\n\nAccuracy score: " + classifier_accuracy)
    elif selected_classifier == 'Multinomial Naive Bayes':
        classifier_description_textbox.config(state=tk.NORMAL)
        classifier_description_textbox.delete('1.0', tk.END)
        classifier_accuracy = str(
            best_models_df.loc[best_models_df['model'] == selected_classifier, 'accuracy'].values[0])
        classifier_description_textbox.insert(tk.END,
                                              "Naive Bayes is a simple probabilistic classifier based on applying "
                                              "Bayes' theorem with strong (naive) independence assumptions between "
                                              "the features. "
                                              "\n\nAccuracy score: " + classifier_accuracy)
    elif selected_classifier == 'Random Forest':
        classifier_description_textbox.config(state=tk.NORMAL)
        classifier_description_textbox.delete('1.0', tk.END)
        classifier_accuracy = str(
            best_models_df.loc[best_models_df['model'] == selected_classifier, 'accuracy'].values[0])
        classifier_description_textbox.insert(tk.END,
                                              "Random forest is a supervised learning algorithm. "
                                              "It can be used both for classification and regression problems. "
                                              "It is a tree-based algorithm that uses multiple decision trees to "
                                              "predict the output. "
                                              "\n\nAccuracy score: " + classifier_accuracy)
    elif selected_classifier == 'Support Vector Machine':
        classifier_description_textbox.config(state=tk.NORMAL)
        classifier_description_textbox.delete('1.0', tk.END)
        classifier_accuracy = str(
            best_models_df.loc[best_models_df['model'] == selected_classifier, 'accuracy'].values[0])
        classifier_description_textbox.insert(tk.END,
                                              "SVM are a set of supervised learning "
                                              "methods used for classification, regression and outliers detection. "
                                              "It uses a subset of training points in the decision function "
                                              "(called support vectors). "
                                              " \n\nAccuracy score: " + classifier_accuracy)
    elif selected_classifier == 'Multi-layer Perceptron':
        classifier_description_textbox.config(state=tk.NORMAL)
        classifier_description_textbox.delete('1.0', tk.END)
        classifier_accuracy = str(
            best_models_df.loc[best_models_df['model'] == selected_classifier, 'accuracy'].values[0])
        classifier_description_textbox.insert(tk.END,
                                              "A multilayer perceptron (MLP) is a class of feedforward "
                                              "artificial neural network. The term MLP is used ambiguously, "
                                              "sometimes loosely to any feedforward ANN, sometimes strictly to "
                                              "refer to networks composed of multiple layers of perceptrons."
                                              "\nAccuracy score: " + classifier_accuracy)
    elif selected_classifier == 'AVG Ensemble':
        classifier_description_textbox.config(state=tk.NORMAL)
        classifier_description_textbox.delete('1.0', tk.END)
        classifier_description_textbox.insert(tk.END,
                                              "Average prediction af all the classifiers above.")

        classifier_description_textbox.config(state=tk.DISABLED)
    # add more elif statements for the remaining classifiers


# the main code to create and run the GUI
window = tk.Tk()
window.title("Email Spam Classifier")
window.geometry("535x700")
window.configure(bg="#c0dffa")
window.resizable(False, False)

# add a label with text "Enter your email:"
email_label = tk.Label(text="Enter your email:", font=("Helvetica", 12), bd=0, bg="#c0dffa")
email_label.grid(row=0, column=0, padx=40, pady=5, columnspan=2, sticky='nsew')

# create the text box for entering the email
email_textbox = tk.Text(window, height=8, width=40, font=("Helvetica", 11), relief=tk.GROOVE)
email_textbox.grid(row=1, column=0, padx=40, pady=5, columnspan=2, sticky='nsew')
# add initial text to the text box
email_textbox.insert(tk.END, "Paste email here")
# when the text box is clicked, clear the text
email_textbox.bind("<Button-1>", lambda event: email_textbox.delete("1.0", "end-1c"))

preprocessed_email_label = tk.Label(text="This is the Preprocessed email:", font=("Helvetica", 12), bg="#c0dffa")
preprocessed_email_label.grid(row=2, column=0, padx=40, pady=5, columnspan=2, sticky='nsew')

# create the text box for displaying the preprocessed email
preprocessed_email_textbox = tk.Text(window, height=8, width=40, font=("Helvetica", 11), relief=tk.GROOVE,
                                     state=tk.DISABLED)
preprocessed_email_textbox.grid(row=3, column=0, padx=40, pady=5, columnspan=2, sticky='nsew')
# add initial text to the text box
preprocessed_email_textbox.insert(tk.END, "")

# add label for the classifier listbox
classifier_label = tk.Label(text="Select a classifier:", font=("Helvetica", 12), bg="#c0dffa")
classifier_label.grid(row=4, column=0, padx=40, pady=5, columnspan=2, sticky='nsew')

# create the listbox for selecting the classifier
classifiers = ['Decision Tree', 'KNN', 'Logistic Regression', 'Multi-layer Perceptron', 'Multinomial Naive Bayes',
               'Random Forest', 'Support Vector Machine', 'AVG Ensemble']
classifier_listbox = tk.Listbox(window, selectmode="single", width=30, height=8, font=("Helvetica", 11),
                                relief=tk.GROOVE)
# bind the listbox to the <<ListboxSelect>> event
classifier_listbox.bind("<<ListboxSelect>>", update_classifier_description)
for classifier in classifiers:
    classifier_listbox.insert(tk.END, classifier)
classifier_listbox.grid(row=5, column=0, padx=10, pady=5)

# add a small text box for displaying the classifier description and accuracy score when a classifier is selected
# the text box should be in right side of the listbox
classifier_description_textbox = tk.Text(window, height=8, width=30, font=("Helvetica", 9), relief=tk.GROOVE,
                                         state=tk.DISABLED)
classifier_description_textbox.grid(row=5, column=1, padx=10, pady=5)

# add label for the classifier listbox
dataset_label = tk.Label(text="Classifier trained on dataset:", font=("Helvetica", 12), bg="#c0dffa")
dataset_label.grid(row=6, column=0, pady=5)

# add a 3 option combo box for selecting the dataset size: 100, 500, 1000
dataset_name = tk.StringVar()
dataset_name.set('emails1')
dataset_name_combobox = ttk.Combobox(window, textvariable=dataset_name,
                                     values=['emails1', 'emails2', 'emails1+2'], width=10,
                                     font=("Helvetica", 11), state="readonly")
dataset_name_combobox.grid(row=6, column=1, pady=5)

dataset_size_label = tk.Label(text="With a dataset size:", font=("Helvetica", 12), bg="#c0dffa")
dataset_size_label.grid(row=7, column=0, pady=5)
# add a 3 option combo box for selecting the dataset size: 100, 500, 1000
dataset_size = tk.StringVar()
dataset_size.set("100")
dataset_size_combobox = ttk.Combobox(window, textvariable=dataset_size, values=["100", "500", "1000"], width=10,
                                     font=("Helvetica", 11), state="readonly")
dataset_size_combobox.grid(row=7, column=1, pady=5)

classify_button = tk.Button(window, text="Classify Email with Best Classifier", command=classify_email_gui,
                            font=("Helvetica", 12), bg="#38a1fc", fg="white", relief=tk.RAISED,
                            width=30)
classify_button.grid(row=8, column=0, pady=5, padx=10, columnspan=2, sticky='nsew')

# add label (c) Ron Keinan 2023
c_label = tk.Label(text="(c) Ron Keinan 2023", font=("Helvetica", 8), bg="#c0dffa")
c_label.grid(row=9, column=0, pady=2, columnspan=2, sticky='nsew')

# run the GUI, open in the center of the screen: width and height of the screen
window.update_idletasks()
width = window.winfo_screenwidth()
height = window.winfo_screenheight()
x = (width / 2) - (window.winfo_width() / 2)
y = (height / 2) - (window.winfo_height() / 2)
window.geometry("+%d+%d" % (x, y))
window.mainloop()
