package net.parasec.nn.training;

import net.parasec.nn.logging.Logger;
import net.parasec.nn.network.ANN;
import net.parasec.nn.util.MathUtil;

/**
 * train the network.
 */
public final class Trainer {
  private static final Logger LOG = Logger.getLogger(Trainer.class);
 
  /**
   * stochastic backpropagation with test/holdback set.
   *
   * note that supplied ANN is optimised in place.
   * if using a test validation subset, ann will be initialised with weights
   * yielding best score on test set.
   */  
  public static TrainingReport train(final ANN ann, final Data data, 
      final int epochs, final double learningRate, final double momentum) {
    int bestEpoch = 0;
    double testingAverageError = 0;
    double testingMinError = 0;
    double testingMaxError = 0;
    
    final int datasetLen = data.trainingSize();
    double lowestError = Double.MAX_VALUE;
    double[][][] bestNetwork = null;

    final double[] trainingError = new double[epochs];
    final double[] testingError = new double[epochs];

    // good practice to randomise the training data. -- neural network should 
    // generalise, not remember.
    // note that this step was performed before each epoch. moved it here since
    // it is probably ok to do this step once. otherwise the training mse 
    // oscillates quite wildly which is amplified in the test subset. 
    data.randomise();

    for(int i = 0; i < epochs; i++) {
      double trainingSum = 0;

      // for each training instance
      for(final TrainingInstance trainingInstance : data.getTrainingData()) {
        final double[] inputVector = trainingInstance.getInputVector();
        final double[] outputVector = trainingInstance.getOutputVector();

        // feed-forward.
        final double[] networkOutput = ann.feedForward(inputVector);

        // backpropagate.
        ann.backPropagateError(outputVector, learningRate, momentum);

	// sum of squares (could be any metric.)
        trainingSum += networkError(networkOutput, outputVector);
      }

      // training root mean square error for this epoch. 
      final double trainingMse = MathUtil.fastSqrt(trainingSum/datasetLen);
      trainingError[i] = trainingMse;

      if(data.testSize() > 0) {
        double testingSum = 0;
        for(final TrainingInstance testingInstance: data.getTestData()) {
          final double[] networkOutput 
              = ann.feedForward(testingInstance.getInputVector());
          testingSum 
              += networkError(networkOutput, testingInstance.getOutputVector());
        }

	// testing root mean square error for this epoch
        // if an improvement, save epoch# and network weights.
        //
        // note: could implement early stopping here.
        //
        final double testingMse = MathUtil.fastSqrt(testingSum/data.testSize());
        if(testingMse < lowestError) {
          bestEpoch = i;
          lowestError = testingMse;
          bestNetwork = ann.cloneWeights();
        }
        testingError[i] = testingMse;

        LOG.info("epoch = " + i + " training rmse = " + 
            String.format("%.10f", trainingMse) + " testing rmse = " +
            String.format("%.10f", testingMse));
      } else {
        LOG.info("epoch = " + i + " rmse = " + 
            String.format("%.10f", trainingMse));
      }
    } // end epochs.

    // if training with validation set.
    if(data.testSize() > 0 && bestNetwork != null) {

      // initialise the NN with the weights yielding the best score on the
      // validation set (best generalisation)
      ann.initialiseWeights(bestNetwork);

      // some validation stats.
      double sum = 0;
      double min = 1;
      double max = 0;
      int i = 0;
      for(final TrainingInstance testingInstance: data.getTestData()) {
        final double[] networkOutput 
            = ann.feedForward(testingInstance.getInputVector());
        final double[] desiredOutput 
            = testingInstance.getOutputVector();
        final int len = networkOutput.length;
        i += len;
        for(int j = 0; j < len; j++) {
          final double diff = Math.abs(networkOutput[j]-desiredOutput[j]);
          sum += diff;
          if(diff < min)
            min = diff;
          if(diff > max)
            max = diff;
        }
      }
      testingAverageError = sum/i;
      testingMinError = min;
      testingMaxError = max;
    } else {
      bestEpoch = epochs-1;
    }
    return new TrainingReport(bestEpoch+1, 
        trainingError[trainingError.length-1], 
        lowestError, testingAverageError, testingMinError, testingMaxError, 
        trainingError, testingError);
  }

  /**
   * sum of squares
   */ 
  private static double networkError(final double[] output, 
      final double[] desiredOutput) {
    double sum = 0;
    for(int i = 0, len = output.length; i < len; i++) {
      final double diff = desiredOutput[i]-output[i];
      sum += diff*diff;
    } 
    return sum;
  }

}

