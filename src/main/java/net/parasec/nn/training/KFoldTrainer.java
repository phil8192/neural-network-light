package net.parasec.nn.training;

import net.parasec.nn.logging.Logger;
import net.parasec.nn.network.ANN;

import java.util.concurrent.Executors;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.List;
import java.util.Random;
import java.util.Stack;

/**
 * train network with k-fold cross validation.
 * see: T. Mitchell. Machine Learning. p112.
 *
 * this implementation trains networks concurrently.
 */
public final class KFoldTrainer {
  private static final Logger LOG = Logger.getLogger(KFoldTrainer.class);

  /**
   * internal: this just encapsulates a training-task: training over a training
   * set + evaluation over a test set.
   */ 
  private final class TrainingTask {
    private final Data data;
    private final int resultIndex;       

    public TrainingTask(final Data data, final int resultIndex) {
      this.data = data;
      this.resultIndex = resultIndex;
    }

    public Data getData() {
      return data;
    }

    public int getResultIndex() {
      return resultIndex;
    }
  }        

  private final double minRandomWeight;
  private final double maxRandomWeight; 
  private final int[] structure;
  private final int maxEpochs;
  private final Random random;

  private final int k;


  public KFoldTrainer(final Random random, final double minRandomWeight, 
      final double maxRandomWeight, final int[] structure, 
      final int maxEpochs, final int k) {
    this.random = random;
    this.minRandomWeight = minRandomWeight;
    this.maxRandomWeight = maxRandomWeight;
    this.structure = structure;
    this.maxEpochs = maxEpochs;
    this.k = k;
  }

  /**
   * --- optimise the number of epochs needed to converge ---
   * note that each partition reserved for test set must be unique.
   * 1. split training data into N test partitions. 
   * 2. for each partition, training data = data - partition N.
   * 3. train neural net on this data and record epoch for lowest error on 
   *    partition N of this data set.
   * 4. train network on all data, with no partitions to average epoch from 
   *    step (3).
   * 5. return neural net.
   */
  public ANN train(final Data data, final double learningRate, 
      final double momentum) {
    //final int numberOfPartitions = (int) Math.round(data.trainingSize()/70);
    final int numberOfPartitions = k;
    final List<List<TrainingInstance>> partitions 
        = data.partition(numberOfPartitions);
    LOG.info("training. " + data.trainingSize() + " / number of partitions = " + 
        numberOfPartitions);
    int i = 1;
    int j = 0;
    for(final List<TrainingInstance> p : partitions) {
      final int size = p.size();
      LOG.info("training. partition " + i + " size = " + size);
      j += size;
      i++;
    }
    LOG.info("training. total = " + j);
    final List<Data> dataPartitions = data.dataPartitions(partitions);
    LOG.info("training. data partitions = " + dataPartitions.size());
    for(final Data d : dataPartitions)
      LOG.info("training. " + d.trainingSize() + " | " + d.testSize());
    final BlockingQueue<TrainingTask> queue // queue for threads to wait on
        = new LinkedBlockingQueue<TrainingTask>();
    final CountDownLatch signal = new CountDownLatch(dataPartitions.size());
    // training task results
    final int[] total = new int[dataPartitions.size()];
    // 8 threads.
    final Stack<Thread> threads = new Stack<Thread>();
    for(int k = 0; k < 8; k++) {
      final Thread t = Executors.defaultThreadFactory().newThread(new Runnable() {
        public final void run() {
          try {
            for(;;) {
              final TrainingTask tt = queue.take();
              LOG.info("training. " + Thread.currentThread().toString() + 
                  " got task from queue...");
              final ANN ann = new ANN(minRandomWeight, maxRandomWeight, 
                  structure, random);
              final TrainingReport report = Trainer.train(ann, tt.getData(), 
                  maxEpochs, learningRate, momentum);
              total[tt.getResultIndex()] = report.getBestEpoch();
              LOG.info("training. " + Thread.currentThread().toString() + 
                  " -> " + report.toString() );
              signal.countDown();
            }
          } catch(final InterruptedException ie) {
            Thread.currentThread().interrupt();
          }
        }
      });
      threads.push(t);
      t.start();
    }
    // put training tasks on queue
    i = 0;
    for(final Data d : dataPartitions)
      queue.offer(new TrainingTask(d, i++));
    // wait for tasks to be complete
    try {
      LOG.info("waiting for k-fold tasks to complete");
      signal.await();
      LOG.info("k-fold tasks complete");
      // kill'em
      while(!threads.isEmpty())
        threads.pop().interrupt();
    } catch(final InterruptedException ie) {
      Thread.currentThread().interrupt();
    }
    // tasks complete. get average epochs and train final network.
    int totalSum = 0;
    for(i = 0; i < total.length; i++)
      totalSum += total[i];
    final int averageBest 
       = (int) Math.round(totalSum/(double) dataPartitions.size());
    LOG.info("training average best = " + averageBest);
    final ANN ann 
        = new ANN(minRandomWeight, maxRandomWeight, structure, random);
    final TrainingReport report 
        = Trainer.train(ann, data, averageBest, learningRate, momentum);
    LOG.info("training complete. " + report);
    return ann;
  }
}

