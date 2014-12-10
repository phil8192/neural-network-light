package net.parasec.nn.training;

import net.parasec.nn.logging.Logger;
import net.parasec.nn.network.ANN;

/**
 * train the network.
 */
public final class Trainer {
  private static final Logger LOG = Logger.getLogger(Trainer.class);
 
  /**
   * stochastic backpropagation with test/holdback set.
   */  
  public static TrainingReport train(final ANN ann, final Data data, 
      final int epochs, final double learningRate, final double momentum) {
    int bestEpoch = 0;
    double trainingMSE = 0;
    double testingMSE = 0;
    double testingAverageError = 0;
    double testingMinError = 0;
    double testingMaxError = 0;
    
    final int datasetLen = data.trainingSize();
    double lowestError = Double.MAX_VALUE;
    double[][][] bestNetwork = null;

    final double[] trainingError = new double[epochs];
    final double[] testingError = new double[epochs];

    for(int i = 0; i < epochs; i++) {
      // good practice to randomise the training data before each epoch
      // -- neural network should generalise, not remember.
      data.randomise();
      double trainingSum = 0;
      for(final TrainingInstance trainingInstance : data.getTrainingData()) {
        final double[] inputVector = trainingInstance.getInputVector();
        final double[] outputVector = trainingInstance.getOutputVector();
        final double[] networkOutput = ann.feedForward(inputVector);
        ann.backPropagateError(outputVector, learningRate, momentum);
        trainingSum += networkError(networkOutput, outputVector);
      }
      trainingMSE = Math.sqrt(trainingSum/datasetLen);
      trainingError[i] = trainingMSE;
      double testingMse = Double.MAX_VALUE;
      if(data.testSize() > 0) {
        trainingSum = 0;
        for(final TrainingInstance testingInstance: data.getTestData()) {
          final double[] networkOutput 
              = ann.feedForward(testingInstance.getInputVector());
          trainingSum 
              += networkError(networkOutput, testingInstance.getOutputVector());
        }
        testingMse = Math.sqrt(trainingSum/data.testSize());
        if(testingMse < lowestError) {
          bestEpoch = i+1;
          lowestError = testingMse;
          bestNetwork = ann.cloneWeights();
        }
        testingError[i] = testingMse;
      }
    } // end epochs.
    if(data.testSize() > 0 && bestNetwork != null) {
      ann.initialiseWeights(bestNetwork);
      testingMSE = lowestError;
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
      bestEpoch = epochs;
    }
    return new TrainingReport(bestEpoch, trainingMSE, testingMSE, 
        testingAverageError, testingMinError, testingMaxError, trainingError,
        testingError);
  }

  /**
   * sum of squares
   */ 
  private static double networkError(final double[] output, 
      final double[] desiredOutput) {
    double sum = 0;
    for(int i = output.length; --i>= 0; )
      sum += Math.pow(desiredOutput[i]-output[i], 2);
    return sum;
  }

}

