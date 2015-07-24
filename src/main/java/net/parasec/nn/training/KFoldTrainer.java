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
   * train K neural networks on K different splits of the data.
   *
   * each split consists of a training set and a validation set.
   * in k-folding, each training+validation combination is unique, such that
   * _all_ of the data will be used for validation.
   *
   * the purpose of k-folding is to determine an expectation of generalisation
   * accuracy for the model. in the context of neural networks; given a fixed
   * structure (the model), the purpose is to see how well the model can be 
   * trained.
   *
   * this implementation returns a set of statistics for each fold along with
   * a set of trained networks. the returned networks are those which yield
   * the best score on the fold validation data: these could potentially be
   * used in an ensemble.
   */
  public KFoldResults[] train(final Data data, final double learningRate,
      final double momentum) {     
    final KFoldResults[] res = new KFoldResults[k];
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

    // queue for threads to wait on
    final BlockingQueue<TrainingTask> queue
        = new LinkedBlockingQueue<TrainingTask>();
    final CountDownLatch signal = new CountDownLatch(dataPartitions.size());

    // training task results
    final Stack<Thread> threads = new Stack<Thread>();
    final int cores = Runtime.getRuntime().availableProcessors();
    LOG.info("using " + cores + " threads.");
    for(int k = 0; k < cores; k++) {
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
              res[tt.getResultIndex()] = new KFoldResults(ann, report);
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
      // kill'em..
      // kill'em all!@#$%!@%
      while(!threads.isEmpty())
        threads.pop().interrupt();
    } catch(final InterruptedException ie) {
      Thread.currentThread().interrupt();
    }
    return res;
  }
}

