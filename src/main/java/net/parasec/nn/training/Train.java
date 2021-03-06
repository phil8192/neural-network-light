package net.parasec.nn.training;

import net.parasec.nn.logging.Logger;
import net.parasec.nn.logging.Report;
import net.parasec.nn.network.ANN;
import net.parasec.nn.util.IO;

import java.util.List;
import java.util.Random;

/**
 * initialise and train a neural network.
 */
public final class Train {
  private final static Logger LOG = Logger.getLogger(Train.class);

  public static void main(String[] args) {

    // dataset args.
    final String file = args[0];
    final int outputLength = Integer.parseInt(args[1]);
    final double holdbackRatio = Double.parseDouble(args[2]);
    final int k = Integer.parseInt(args[3]);

    // training args.
    final double minRandomWeight = Double.parseDouble(args[4]);
    final double maxRandomWeight = Double.parseDouble(args[5]); 
    final double learningRate = Double.parseDouble(args[6]);
    final double momentum = Double.parseDouble(args[7]);
    final int maxEpochs = Integer.parseInt(args[8]);

    // post processing
    final String modelOutput = args[9];

    // prng used throughout training
    final Random prng = new Random();

    // load the data from specified csv.   
    final DataLoader dl = new DataLoader();
    final List<TrainingInstance> tiList = dl.loadCsv(file, outputLength);

    // construct dataset.
    // dataset will be split into a training and test validation subset.
    // the test validation subset size is holdbackRatio*total_data_size.
    final Data data = new Data(prng);

    data.getTrainingData().addAll(tiList);
    if(holdbackRatio > 0)
      data.split(holdbackRatio);

    // network hidden layer structure args.
    // (number of input and output nodes determined from dataset outputLength).
    // remaining args are the number of hidden nodes in each layer of the
    // network.
    final int inputNodes = tiList.get(0).getInputVector().length;
    final int hiddenLayers = args.length-10;
    final int outputNodes = outputLength;
    final int[] structure = new int[2+hiddenLayers]; 
    structure[0] = inputNodes;
    for(int i = 10, len = args.length; i < len; i++)
      structure[i-9] = Integer.parseInt(args[i]);
    structure[structure.length-1] = outputNodes;

    LOG.info("min_rw = " + minRandomWeight +
             " max_rw = " + maxRandomWeight +
             " lr = " + learningRate +
             " mo = " + momentum);       

    // train the network.
    // results in dumping network weights and training errors to disk.  
 
    // if no k-folding, train as normal
    if(k <= 1) {
      final ANN ann = new ANN(minRandomWeight, maxRandomWeight, structure, prng);
      final long l = System.currentTimeMillis();      
      final TrainingReport report = Trainer.train(ann, data, maxEpochs, 
          learningRate, momentum);
      LOG.info("training complete. " + report);
      LOG.info("training took " + (System.currentTimeMillis()-l) + "ms.");
      Report.dump(report, modelOutput);
      IO.dumpWeights(ann.getWeights(), modelOutput + "/weights.bin");
    } else {
      // use the k-fold-trainer.
      final KFoldTrainer kft = new KFoldTrainer(prng, minRandomWeight, 
          maxRandomWeight, structure, maxEpochs, k);
      //ann = kft.train(data, learningRate, momentum);
      final KFoldResults[] kfr = kft.train(data, learningRate, momentum);
      for(int i = 1, len = kfr.length; i <= len; i++) {
        final KFoldResults _kfr = kfr[i-1];
        final TrainingReport tr = _kfr.getTrainingReport(); 
        LOG.info("K-Fold " + String.format("%02d", i) + " " + tr);
        Report.dump(tr, modelOutput, "errors_" +
            String.format("%02d", i) + ".csv");
        IO.dumpWeights(_kfr.getANN().getWeights(), modelOutput + "/weights_" + 
            String.format("%02d", i) + ".bin");
      }
    }
  }
}

