# Usage 
Using Naive Baye's model to classify emails as SPAM or HAM


1. nblearn.py <input_dir> [-d | --dataset training_data_percentage]
    
    The script reads files specified in the input directory and creates a naive bayes model that can be used to classify emails
    as SPAM or HAM. The model is output as nbmodel.txt in the current working directory. Additionally, a "-d" parameter can be specified to this script, which allows the user to choose what percentage of training data should be used for generating the model.
    for. e.g. nblearn.py "/home/jarvis/mydata/" -d 10 - will generate a model trained on only 10% of the data
    
    For classifiying data as HAM or SPAM, the input directory should contain classified data in format <fileprefx>_spam.txt or <fileprefx>_ham.txt . The model created with this code is known to give 98% precision for SPAM data in test environments.
    
2. nbclassify.py <idir> [-m | --model <filename>] 
    
    The script uses the model specified by --model option or looks for nbmodel.txt file in the current directory for classifiying test data. The idir option specifies the input directory for the test data. The statistics of the classification (precision/ recall/ accurancy and F1 score) are output at the end of the script run. The script generates nboutput.txt which containes tags for each
    processed file in the format <class_name> <file_name_with_path>
    
    
    
