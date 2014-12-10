package net.parasec.nn.training;

import net.parasec.nn.logging.Logger;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * represents the data/subsets used in supervised learning.
 */ 
public final class Data {
  private static final Logger LOG = Logger.getLogger(Data.class);  

  private final List<TrainingInstance> trainingData = new ArrayList<TrainingInstance>();
  private final List<TrainingInstance> testData = new ArrayList<TrainingInstance>();
  private final Random random;


  public Data(final Random random) {
    this.random = random;
  }

  /**
   * training data subset.
   */ 
  public List<TrainingInstance> getTrainingData() {
    return trainingData;
  }

  /**
   * testing data subset.
   */ 
  public List<TrainingInstance> getTestData() {
    return testData;
  }

  /**
   * given a list of partition lists, create a data object for each partition,
   * where the test-set is the list of training instances and the training set
   * is the list of all other training instances:
   * [a,b,c,d,e,f] test sets: [a,b] [c,d] [e,f],
   * partitions: 1: training [c,d,e,f] testing: [a,b]
   *             2: training [a,b,e,f] testing: [c,d]
   *             3: training [a,b,c,d] testing: [e,f] 
   */ 
  public List<Data> dataPartitions(final List<List<TrainingInstance>> 
      partitions) {
    final List<Data> data = new ArrayList<Data>();
    for(final List<TrainingInstance> partition : partitions) {
      final Data d = new Data(random);
      final List<TrainingInstance> training = d.getTrainingData();
      final List<TrainingInstance> testing = d.getTestData();
      testing.addAll(partition);
      for(final List<TrainingInstance> otherPartition : partitions)
        if(!partition.equals(otherPartition)) 
          training.addAll(otherPartition);
      data.add(d);
    }
    return data;
  }

  /**
   * shuffle, and return a list of n partitions.
   * each partition size is #training_size/#partitions.
   */ 
  public List<List<TrainingInstance>> partition(final int partitions) {
    final List<List<TrainingInstance>> ret 
        = new ArrayList<List<TrainingInstance>>();
    final int partitionSize 
        = (int) Math.round(trainingData.size()/(double) partitions);
    final List<TrainingInstance> copy 
        = new ArrayList<TrainingInstance>(trainingData);
    Collections.shuffle(copy, random);
    int k = 0;	
    for(int i = 0; i < partitions-1; i++) {
      final List<TrainingInstance> partition 
          = new ArrayList<TrainingInstance>();
      ret.add(partition);
      for(int j = 0; j < partitionSize; j++)
	partition.add(copy.get(k++));
    }				
    final List<TrainingInstance> partition = new ArrayList<TrainingInstance>();
    ret.add(partition);
    for(int len = copy.size(); k < len; k++) 
      partition.add(copy.get(k));	
    return ret;
  }

  /**
   * split data into training and test subsets. 
   * &gt; ratio = test data. 0.2 = 20% test data.
   * initially all data is training data.
   */
  public void split(final double ratio) {
    randomise();
    final int tdSize = trainingData.size();
    LOG.info("splitting data into training/testing subsets");
    for(int i = (int) Math.round((ratio * (double) tdSize)); --i >= 0; )
      testData.add(trainingData.remove(trainingData.size()-1));
  }

  /**
   * randomise the order of training data.
   */ 
  public void randomise() {
    LOG.debug("randomising training data");
    Collections.shuffle(trainingData, random);
  }

  public int trainingSize() {
    return trainingData.size();
  }
 
  public int testSize() {
    return testData.size();
  }

  public void print() {
    LOG.info("training data");
    LOG.info("=============");
    for(final TrainingInstance ti : trainingData)
      LOG.info(ti);
    LOG.info("testing data");
    LOG.info("============");	
    for(final TrainingInstance ti : testData)
      LOG.info(ti);
  }
}

