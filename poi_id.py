#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import cPickle
import numpy as np

from tester import dump_classifier_and_data

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.feature_selection import SelectKBest


########################################################################
### Task 1: Select what features you'll use.
########################################################################

# features_list is a list of strings, each of which is a feature name. The first feature is "poi".
features_list = ["poi","salary", "deferral_payments", "total_payments", "loan_advances", "bonus", "restricted_stock_deferred",\
				 "deferred_income", "total_stock_value", "expenses", "exercised_stock_options", "other", "long_term_incentive",\
				 "restricted_stock", "director_fees","to_messages", "from_poi_to_this_person", "from_messages",\
				 "from_this_person_to_poi", "shared_receipt_with_poi" ]

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

########################################################################
### Task 3: Helper functions
########################################################################

# Helper function to create new fraction_from_poi & fraction_to_poi features.
# Credit: www.udacity.com/course/intro-to-machine-learning--ud120 course developers
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person that are from/to a POI
   	"""
    fraction = 0.
    if poi_messages == "NaN" or all_messages == "NaN":
        fraction = 0.
    else:
        fraction = float(poi_messages) / all_messages
    return fraction


# Helper function to separate features to target list and remaining features list
# Credit: www.udacity.com/course/intro-to-machine-learning--ud120 course developers
def targetFeatureSplit( data ):
    """ 
        given a numpy array, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists
    """
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features
 

# Helper functions to convert data dictionery to numpy array of features
# Credit: www.udacity.com/course/intro-to-machine-learning--ud120 course developers
def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """
    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == "poi":
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.

        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)

########################################################################
## Task 2: Remove outliers
########################################################################

# Analyzing the data set:
if False:
	print len(data_dict.items())
	print len(data_dict["CHAN RONNIE"])
	print data_dict["CHAN RONNIE"]

	for key_d, values in data_dict.items():
		value_sum = 0
		for key_point, value in values.items():
			if  type(value) == int:
				value_sum += value
		if value_sum == 0:		
			print "**",key_d, value_sum	
			

# removing not relevant data points to the analysis
data_dict.pop( "TOTAL", 0 )
data_dict.pop( "THE TRAVEL AGENCY IN THE PARK", 0)


# For possible removal of outlier data point:
if False:
	data_dict.pop( "LAY KENNETH L",0)

# Plotting features to test the outliers
if False:
	label_x = features_list[1] + ', K'
	label_y = features_list[3] + ', K'


	data = featureFormat(data_dict, features_list)	

	for point in data:
	    point_x = point[1] / 1000
	    point_y = point[3] / 1000
	    plt.scatter( point_x, point_y, color = "r", alpha=0.7)   


	plt.xlabel(label_x)
	plt.ylabel(label_y)

	plt.ylim(-500, 18000)
	plt.xlim(-100, 1100)


	plt.title('Salary vs. Total Payments (K, in thousands)')

	plt.show()


########################################################################
### Task 4: Create new feature(s)
########################################################################

# Adding new features for each data point to data_dict 
# Credit: www.udacity.com/course/intro-to-machine-learning--ud120 course developers
for name in data_dict:
    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

# Adding new features to features list for training and/or testing
# and removed the features used to calculate these new ones.
if True:
	features_list.append("fraction_from_poi")
	features_list.append("fraction_to_poi")

	features_list.remove( "from_poi_to_this_person")
	features_list.remove( "from_this_person_to_poi")
	features_list.remove( "to_messages" )
	features_list.remove( "from_messages" )



# Removed salary, deferral_payments, bonus, expenses, loan_advances, other,
# deferred_income, long_term_incentive features, since total_payments are sum of these
if False:
	for name in data_dict:
	    data_point = data_dict[name]

	    total_payments = 0

	    if data_point['exercised_stock_options'] != 'NaN':
	        total_payments += data_point['exercised_stock_options']
	    if data_point['restricted_stock'] != 'NaN':
	        total_payments += data_point['restricted_stock']
	       

	    print name, total_payments, data_point['total_stock_value']


 
# Testing new features: print a single entry in data_dict
if targetFeatureSplit:
	if False:
		print "features_list: {}".format(features_list)
		print "Rest data dictionary: {}".format(data_dict["COLWELL WESLEY"])

# Store to my_dataset for easy export below.
my_dataset = data_dict


# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Testing new features: plotting
if False:
	label_x = features_list[-1]
	label_y = features_list[-2]

	for point in features:
	    point_x = point[-1]
	    point_y = point[-2]
	    plt.scatter( point_x, point_y, color = "b")   


	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.ylim(-0.01, 0.23)
	plt.xlim(-0.05, 1.05)

	plt.title('fraction_from_poi vs. fraction_to_poi')

	plt.show()


########################################################################
### Task 5: Try a variety of classifiers
########################################################################

features_train, features_test, labels_train, labels_test = \
		cross_validation.train_test_split(features, labels, test_size=0.35, random_state=42)


# Compute the metrics of Decision Tree Classifier
if False:

	clf = DecisionTreeClassifier( class_weight=None, criterion="entropy", max_depth=None,
	            max_features="auto", max_leaf_nodes=None, min_samples_leaf=1,
	            min_samples_split=3, min_weight_fraction_leaf=0.0,
	            presort=False, random_state=42, splitter="best" )

	fit_classifier = clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)

	accuracy = accuracy_score(labels_test, pred)
	recall_sc = recall_score(labels_test, pred)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)

 	print "Decision Tree accuracy: {}".format( round( accuracy, 3 ) )
 	print "Decision Tree recall_score: {}".format( round( recall_sc, 3) )
 	print "Decision Tree precision_score: {}".format( round( precision_sc, 3) )
 	print "Decision Tree f1_score: {}\n".format( round( f1_sc, 3) )

# Compute the metrics of GaussianNB classifier
if False:
	clf = GaussianNB()
	fit_classifier = clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)

	accuracy = accuracy_score(labels_test, pred)
	recall_sc = recall_score(labels_test, pred)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)

	print "GaussianNB accuracy: {}".format( accuracy)
	print "GaussianNB recall_score: {}".format(recall_sc)
	print "GaussianNB precision_score: {}".format(precision_sc)
	print "GaussianNB f1_score: {}\n".format(f1_sc)

# Compute the metrics of AdaBoostClassifier
if False:
	clf = AdaBoostClassifier(algorithm="SAMME.R",n_estimators=300,random_state=0)
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)

	acc = accuracy_score(pred, labels_test)
	recall_sc = recall_score(labels_test, pred)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)

	print "AdaBoostClassifier accuracy: {}\n".format(acc)
	print "AdaBoostClassifier recall_score: {}".format(recall_sc)
	print "AdaBoostClassifier precision_score: {}".format(precision_sc)
	print "AdaBoostClassifier f1_score: {}\n".format(f1_sc)

# Compute the metrics of RandomForestClassifier
if False:
	clf = RandomForestClassifier(n_estimators=500,min_samples_split=2,max_depth=20,min_samples_leaf=2)
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)

	acc = accuracy_score(pred, labels_test)
	recall_sc = recall_score(labels_test, pred)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)
	
	print "RandomForestClassifier accuracy: {}\n".format(acc)
	print "RandomForestClassifier recall_score: {}".format(recall_sc)
	print "RandomForestClassifier precision_score: {}".format(precision_sc)
	print "RandomForestClassifier f1_score: {}\n".format(f1_sc)

# Compute the metrics using Support Vector Classification
if False:

	parameters = {"kernel":['rbf'], "C":[1, 1000],"gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
	svr = SVC()
	clf = GridSearchCV(svr, parameters)

	#clf = SVC(kernel="rbf",C=1000)

	clf.fit(features_train, labels_train) 
	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)
	recall_sc = recall_score(labels_test, pred)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)

	print "SVC accuracy: {}".format( acc)
	print "SVC recall_score: {}".format(recall_sc)	
	print "SVC precision_score: {}".format(precision_sc)
	print "SVC f1_score: {}\n".format(f1_sc)
	print "SVC best parameters: {}\n".format(clf.best_params_)

 


###############################################################################
### Task 6: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 
##############################################################################

features_train, features_test, labels_train, labels_test = \
		cross_validation.train_test_split(features, labels, test_size=0.35, random_state=42)


# Decision Tree
if True:


	#Split features and labels to test/ train subsets
	acc_ave = 0
	recall_sc_ave = 0
	precision_sc_ave = 0
	f1_sc_ave =0
	folds = 4


	kf = cross_validation.KFold(len(labels), folds)

	for train_indices, test_indices in kf:

		# splitting data into train/test subsets
		features_train = [features[i] for i in train_indices]
		features_test = [features[i] for i in test_indices]
		labels_train = [labels[i] for i in train_indices]
		labels_test = [labels[i] for i in test_indices]
		
		# Transforms features by scaling each feature to [0, 1] range
		min_max_scaler = preprocessing.MinMaxScaler()
		features_train_scaled = min_max_scaler.fit_transform(features_train)
		features_test_scaled = min_max_scaler.fit_transform(features_test)


		print features_train_scaled

		# Reduce the feature dimentions using Principal component analysis PCA
		n_components = 5
		pca = PCA(n_components=n_components, whiten=True)
		pca.fit(features_train_scaled)


		# TEST: plot first pca component to test and visualize
		if False:
			pca_first = pca.components_[0]
			pca_second = pca.components_[1]
			print "pca first component: {}".format(pca_first)
			print "***"	

		# Select features according to the k highest scores
		selection = SelectKBest(k=5)

		# Build estimator from PCA and Univariate SelectKBest selection:
		combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

		# Use combined train features to fit and transform train subset:
		features_train_reduced = combined_features.fit( features_train_scaled, labels_train ).transform( features_train_scaled )

		# Use combined test features to transform test subset:
		features_test_reduced = combined_features.transform( features_test_scaled )

		# parameters to be passed to GridSearchCV for cross validation
		parameters  = {"criterion" : ["gini", "entropy"],
					"min_samples_split": [2,3]
	                }

	    # creating classifier            
		svr = DecisionTreeClassifier( class_weight=None, criterion="gini", max_depth=None,
		            max_features="auto", max_leaf_nodes=None, min_samples_leaf=1,
		            min_samples_split=2, min_weight_fraction_leaf=0.0,
		            presort=False, random_state=42, splitter="best" )

		# passing in classifier and the parrameters ????????
		clf = GridSearchCV( svr, parameters )

		# fittig training subset
		clf.fit( features_train_reduced, labels_train ) 

		# passing in test subset to get the prediction coeficient
		pred = clf.predict( features_test_reduced )

		# calculating metrics for the current k selection of the dataset
		acc = accuracy_score( pred, labels_test )
		recall_sc = recall_score( labels_test, pred )
		precision_sc = precision_score( labels_test, pred )
		f1_sc = f1_score( labels_test, pred )

		# calculatting total of the metrics
		acc_ave += acc
		recall_sc_ave += recall_sc
		precision_sc_ave += precision_sc
		f1_sc_ave += f1_sc


	# printing averages of the metrics
	if True:
		print "Decision Tree average accuracy: {}".format( round( acc_ave/ folds, 3) )
		print "Decision Tree average recall_score: {}".format( round( recall_sc_ave / folds,3) )
		print "Decision Tree average precision_score: {}".format( round( precision_sc_ave / folds, 3) )
		print "Decision Tree average f1_score: {}\n".format( round( f1_sc_ave/ folds , 3))	
		print "Decision Tree best parameters : {}".format(clf.best_params_)
	


# GaussianNB
if False:

	#Split features and labels to test/ train subsets
	acc_ave = 0
	recall_sc_ave = 0
	precision_sc_ave = 0
	f1_sc_ave =0
	folds = 5


	kf = cross_validation.KFold(len(labels), folds)

	for train_indices, test_indices in kf:

		# splitting data into train/test subsets
		features_train = [features[i] for i in train_indices]
		features_test = [features[i] for i in test_indices]
		labels_train = [labels[i] for i in train_indices]
		labels_test = [labels[i] for i in test_indices]
		
		# Transforms features by scaling each feature to [0, 1] range
		min_max_scaler = preprocessing.MinMaxScaler()
		features_train_scaled = min_max_scaler.fit_transform(features_train)
		features_test_scaled = min_max_scaler.fit_transform(features_test)

		# Reduce the feature dimentions using Principal component analysis PCA
		n_components = 7
		pca = PCA(n_components=n_components, whiten=True)
		pca.fit(features_train_scaled)


		# TEST: plot first pca component to test and visualize
		if False:
			pca_first = pca.components_[0]
			pca_second = pca.components_[1]
			print "pca first component: {}".format(pca_first)
			print "***"

			# for i, j in zip(features_reduced, features_scaled):
			# 	plt.scatter(pca_first[0] * i[0], pca_first[1] * i[0], color="r")
			# 	plt.scatter(pca_second[0] * i[1], pca_second[1] * i[1], color="y")
			# 	plt.scatter(j[0], j[1], color="b")
			# plt.xlabel("x component")
			# plt.ylabel("y component")	
			# plt.show()	

		# Select features according to the k highest scores
		selection = SelectKBest(k=7)

		# Build estimator from PCA and Univariate SelectKBest selection:
		combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

		# Use combined train features to fit and transform train subset:
		features_train_reduced = combined_features.fit( features_train_scaled, labels_train ).transform( features_train_scaled )

		# Use combined test features to transform test subset:
		features_test_reduced = combined_features.transform( features_test_scaled )


	    # creating classifier            
		clf = GaussianNB()

		# fittig training subset
		clf.fit( features_train_reduced, labels_train ) 

		# passing in test subset to get the prediction coeficient
		pred = clf.predict( features_test_reduced )

		# calculating metrics for the current k selection of the dataset
		acc = accuracy_score( pred, labels_test )
		recall_sc = recall_score( labels_test, pred )
		precision_sc = precision_score( labels_test, pred )
		f1_sc = f1_score( labels_test, pred )

		# calculatting total of the metrics
		acc_ave += acc
		recall_sc_ave += recall_sc
		precision_sc_ave += precision_sc
		f1_sc_ave += f1_sc

	# printing averages of the metrics
	if True:
		print "GaussianNB average accuracy: {}".format( round( acc_ave/ folds, 3) )
		print "GaussianNB average recall_score: {}".format( round( recall_sc_ave / folds,3) )
		print "GaussianNB average precision_score: {}".format( round( precision_sc_ave / folds, 3) )
		print "GaussianNB average f1_score: {}\n".format( round( f1_sc_ave/ folds , 3))	


###############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. poi_id.py can be run on its own and
### generates the necessary .pkl files for validating your results.
###############################################################################
def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

dump_classifier_and_data(clf, my_dataset, features_list)