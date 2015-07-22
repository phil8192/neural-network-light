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
      //data.randomise();
      double trainingSum = 0;
      for(final TrainingInstance trainingInstance : data.getTrainingData()) {
        final double[] inputVector = trainingInstance.getInputVector();
        final double[] outputVector = trainingInstance.getOutputVector();
        final double[] networkOutput = ann.feedForward(inputVector);
/*
        LOG.info("epoch = " + i + 
                 " in = " + java.util.Arrays.toString(inputVector) +
                 " out = " + java.util.Arrays.toString(networkOutput) +
                 " des = " + java.util.Arrays.toString(outputVector));
*/
        ann.backPropagateError(outputVector, learningRate, momentum);
        trainingSum += networkError(networkOutput, outputVector);
      }
      //final double trainingMse = Math.sqrt(trainingSum/datasetLen);
      final double trainingMse = MathUtil.fastSqrt(trainingSum/datasetLen);
      trainingError[i] = trainingMse;
      LOG.info("epoch = " + i + " mse = " + String.format("%.10f", trainingMse));
      if(data.testSize() > 0) {
        double testingSum = 0;
        for(final TrainingInstance testingInstance: data.getTestData()) {
          final double[] networkOutput 
              = ann.feedForward(testingInstance.getInputVector());
          testingSum 
              += networkError(networkOutput, testingInstance.getOutputVector());
        }
        //final double testingMse = Math.sqrt(trainingSum/data.testSize());
        final double testingMse = MathUtil.fastSqrt(testingSum/data.testSize());
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
    return new TrainingReport(bestEpoch, trainingError[trainingError.length-1], 
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

