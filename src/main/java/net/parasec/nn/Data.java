package net.parasec.nn;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

import org.apache.log4j.Logger;


public final class Data {
  private static final Logger LOG = Logger.getLogger(Data.class);  

  private final List<TrainingInstance> trainingData = new ArrayList<TrainingInstance>();
  private final List<TrainingInstance> testData = new ArrayList<TrainingInstance>();
  private final Random random;


  public Data(final Random random) {
    this.random = random;
  }

  public List<TrainingInstance> getTrainingData() {
    return trainingData;
  }

  public List<TrainingInstance> getTestData() {
    return testData;
  }

  public List<Data> dataPartitions(final List<List<TrainingInstance>> 
      partitions) {
    final List<Data> data = new ArrayList<Data>();
    for(final List<TrainingInstance> partition : partitions) {
      final Data d = new Data(random);
      final List<TrainingInstance> training = d.getTrainingData();
      final List<TrainingInstance> testing = d.getTestData();
      testing.addAll( partition );
      for(final List<TrainingInstance> otherPartition : partitions)
        if(!partition.equals(otherPartition)) 
          training.addAll(otherPartition);
      data.add(d);
    }
    return data;
  }

  public List<List<TrainingInstance>> partition(final int partitions) {
    final List<List<TrainingInstance>> ret 
        = new ArrayList<List<TrainingInstance>>();
    final int partitionSize 
        = (int) Math.round(trainingData.size()/(double) partitions);
    final List<TrainingInstance> copy 
        = new ArrayList<TrainingInstance>(trainingData);
    Collections.shuffle( copy, random );
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

  // > ratio = test data. 0.2 = 20% test data.
  public void split(final double ratio) {
    randomise();
    final int tdSize = trainingData.size();
    for(int i = (int) Math.round((ratio * (double) tdSize)); --i >= 0; )
      testData.add(trainingData.remove(trainingData.size()-1));
  }

  public void randomise() {
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

