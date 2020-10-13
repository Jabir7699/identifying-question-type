# QuestionClassification

Problem Description:

PFA of the NLP Task with the following guidelines:

1. Use a learning based approach for solving the task
2. Share the git link along with dependencies and how to run the code

Identify Question Type: Given a question, the aim is to identify the category it belongs to.
The four categories to handle for this assignmentare : Who, What, When, Affirmation(yes/no).
Label any sentence that does not fall in any of the above four as "Unknown" type.
You should come up with data structures to encapsulate these information as well as the code that populates the relevant data.
You can use any machine learning technique and use the attached labelled data set for learning. 
The output should be driven keeping in mind the reply to the question.

Example:
1. What is your name? Type: What
2. When is the show happening? Type: When
3. Is there a cab available for airport? Type: Affirmation
There are ambiguous cases to handle as well like:
What time does the train leave(this looks like a what question but is actually a When type)

For testing, you can also look for datasets on the net. Sample (though the categories are different here): 
http://cogcomp.cs.illinois.edu/Data/QA/QC/train_1000.label

Approach:
1. First preprocessed all the questions and also added bigrams in corpus. 
2. And then Used Doc2Vec implementation from gensim to convert questions to vectors which then fed to our neural Architecture.
3. Created 2 hidden layer Neural Net and computed softmax at end. Used AdamOptimiser to optimise the cost function. This is simple architecture. This model overfits the data which need to be looked into and reduce the overfitting.
4. I have some ideas and need to implement this in future. Use Word2Vec embedding trained on a large dataset and use it to convert questions in vector. Use padding to have a fixed size of input for various length of questions.
5. Then apply Text CNN with filter weights and biases and add maxpool and Relu to add non-linearity to it.
6. Then again use cross_entropy, softmax and AdamOptimiser to train the model.


Instructions:
1. Requirement.txt attached which contains list of python packages used in the project
2. Doc2VecScript1.py is the main script.
3. Dataset "LabelledData.txt" should in the folder in which script is placed.
4. Use command "python Doc2VecScript1.py" to run the script
