package net.parasec.nn.network;

import net.parasec.nn.logging.Logger;
import net.parasec.nn.util.IO;
import net.parasec.nn.util.Util;
import net.parasec.nn.training.DataLoader;
import net.parasec.nn.training.TrainingInstance;

import java.util.List;


public final class Runner {
  private static final Logger LOG = Logger.getLogger(Runner.class);
  public static void main(String[] args) {
    final String weightFile = args[0];
    final String dataFile = args[1];
    final double[][][] weights = IO.loadWeights(weightFile); 
    final ANN ann = new ANN(weights);
    final int outputNodes = weights[weights.length-1].length;
    final DataLoader dl = new DataLoader();
    final List<TrainingInstance> tiList = dl.loadCsv(dataFile, outputNodes);
    for(final TrainingInstance trainingInstance : tiList) { 
      final double[] inputVector = trainingInstance.getInputVector();
      final double[] outputVector = trainingInstance.getOutputVector();
      final double[] networkOutput = ann.feedForward(inputVector);
      LOG.info("in = " + Util.vectorToString(inputVector) + 
               " out = " + Util.vectorToString(outputVector) + 
               " net = " + Util.vectorToString(networkOutput));
    }
  }
}

