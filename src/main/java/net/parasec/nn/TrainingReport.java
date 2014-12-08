package net.parasec.nn;

public final class TrainingReport {
  private final int bestEpoch;
  private final double trainingMSE;
  private final double testingMSE;
  private final double testingAverageError;
  private final double testingMinError;
  private final double testingMaxError;


  public TrainingReport(final int bestEpoch, final double trainingMSE, 
      final double testingMSE, final double testingAverageError, 
      final double testingMinError, final double testingMaxError) {
    this.bestEpoch = bestEpoch;
    this.trainingMSE = trainingMSE;
    this.testingMSE = testingMSE;
    this.testingAverageError = testingAverageError;
    this.testingMinError = testingMinError;
    this.testingMaxError = testingMaxError;
  }

  public int getBestEpoch() {
    return bestEpoch;
  }

  public String toString() {
    return "best epoch = " + bestEpoch + " training mse = " + 
        String.format("%.5f", trainingMSE) + " testing mse = " + 
        String.format("%.5f", testingMSE) + " testing ae = " + 
        String.format("%.2f", testingAverageError) + " min = " + 
        String.format("%.2f", testingMinError) + " max = " + 
        String.format("%.2f", testingMaxError);
  }

}

