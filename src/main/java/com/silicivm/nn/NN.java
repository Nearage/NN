/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */
package com.silicivm.nn;

import com.silicivm.jnn.input.Matrix;
import com.silicivm.jnn.model.Sequential;
import com.silicivm.jnn.model.layer.Dense;
import com.silicivm.jnn.processing.Activation;
import com.silicivm.jnn.processing.Loss;
import com.silicivm.jnn.processing.Model;

/**
 *
 * @author vulgr
 */
public class NN {

    public static void main(String[] args) throws Exception {        
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 200;
        double learningRate = args.length > 1 ? Double.parseDouble(args[1]) : .0002;
        String inputPath = args.length > 2 ? args[2] : "input-400-256.matrix";
        String targetPath = args.length > 3 ? args[3] : "target-400-2.matrix";

        Matrix input = Matrix.load(inputPath);
        Matrix target = Matrix.load(targetPath);

        Model model = new Sequential(
            new Dense(32, Activation.RectifiedLinearUnit),
            new Dense(32, Activation.RectifiedLinearUnit),
            new Dense(2, Activation.Softmax)
        );

        Matrix prediction = model.predict(input);

        model.summary();

        System.out.printf(
            "Pre-training error: %.8f%n%n",
            Loss.MeanSquaredError[0].apply(prediction, target).peek()
        );

        model.train(epochs, input, target, Loss.MeanSquaredError, learningRate);
    }
}
