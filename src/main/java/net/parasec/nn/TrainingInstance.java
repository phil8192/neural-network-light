package net.parasec.nn;

public final class TrainingInstance {
  private final double[] inputVector;
  private final double[] outputVector;

 
  public TrainingInstance(final double[] inputVector, 
      final double[] outputVector) {
    this.inputVector = inputVector;
    this.outputVector = outputVector;
  }

  public double[] getInputVector() {
    return inputVector;
  }

  public double[] getOutputVector() {
    return outputVector;
  }

  public String toString() {
    return Util.vectorToString(inputVector) + " = " + 
        Util.vectorToString(outputVector);
  }
}

