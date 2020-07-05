I did a detailed analysis and found there are two way to complie the keras model. They are listed below

## binary_crossentropy

From given Jupyter book, we are using binary_crossentropy model to complie and find accuracy & loss. Again there are two ways to calculate the accuracy within binary_crossentropy.

[1] `model.evaluate()` is calculating accuracy between sequential predict vs the given predict in cleaned_hm.csv ( as you shown in your screenshot) for the records 20107 which we took for validation



    accuracy = 		Total number of True values in Sequential Predicated category
				---------------------------------------------------------------------------  X 100
						Total number of validation records


[2] Keras `K.mean()` method to calculate the accuracy which will return 95% accuracy as printed in the output. This is what we have used in our script.Let me try to explain the why this way of accuracy calculation methods return different values than model.evaluate()

Keras `K.mean()` method is calculating accuracy using the one-hot encoded / binarise values for both sequential predict and given predict in cleaned_hm.csv using the formula

    K.mean(K.equal(y_val, K.round(sequentialpredict)))


Assume that

	y_val = [[1,0,0],[0,1,0],[0,0,1],[0,1,0]] # from cleaned_hm.csv

	sequentialpredict = [[0.4,0.35,0.35],[0.3,0.4,0.3],[0.1,0.4,0.5],[0.1,0.5,0.4]] # from sequential method prediction output

In the given equation, first `K.round()`, rounds your predictions:

		K.round(sequentialpredict): [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

Then, `K.equal()` checks whether these two results are equal

		K.equal(y_val, K.round(sequentialpredict)) :

		[[1==0,0==0,0==0], [0==0,1==0,0==0], [0==0,0==0,1==0],[0==0,1==0,0==0]]

That results

		[[False, True, True], [True,False,True], [True,True,False], [True,False,True]]

What `K.mean()` does is, simply, sums above result list as if True = 1 and False = 0 and divides all the test row counts.

		binary_accuracy score is 8/12 = 0.66  

In our case, following values been used to derive ~95% accuracy output

Onehot encoded values represent - 20107(validation records) X 7(predicted labels) = 140749
Out of which 133900 turned as true values, so it returned 95.1339% accuracy.

## categorical_crossentropy

On the other hand categorical_accuracy, which defined as follows,

    K.mean(K.cast(K.equal(K.argmax(y_val, axis=-1),K.argmax(sequentialpredict, axis=-1)),K.floatx()))

checks only the highest values of index for y_val and sequentialpredict:

    y_val = [0,1,2,1]

    sequentialpredict = [0,1,2,1]

    [True,True,True,True] => categorical_accuracy = 1

If you use categorical_crossentropy, I recommend you to use categorical_accuracy for precisize results.

In our case, following values been used to derive categorical_crossentropy accuracy output

accuracy value is : 0.706520

PS! - This categorical_crossentropy way is not used in our script but I have tested it locally.
