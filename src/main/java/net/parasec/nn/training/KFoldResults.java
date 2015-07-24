package net.parasec.nn.training;

import net.parasec.nn.network.ANN;

/**
 *
 * results of k-folding
 *
 *
 * for each fold:
 *
 *   - nn with best score (best gernalisation) on validation set  
 *   - training stats (best epoch, lowest errors etc.)
 *
 */
public final class KFoldResults{
  private final ANN ann;
  private final TrainingReport tr;
  public KFoldResults(final ANN ann, final TrainingReport tr){
    this.ann=ann;this.tr=tr;
  }
  public ANN getANN(){return this.ann;}
  public TrainingReport getTrainingReport(){return this.tr;} 
}

