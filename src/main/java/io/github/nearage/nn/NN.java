/*
 * Copyright (C) 2023 Nearage <https://github.com/Nearage>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package io.github.nearage.nn;

import io.github.nearage.jnn.input.Dataset;
import io.github.nearage.jnn.input.Matrix;
import io.github.nearage.jnn.model.Sequential;
import io.github.nearage.jnn.model.layer.Dense;
import io.github.nearage.jnn.processing.Activation;
import io.github.nearage.jnn.processing.Loss;
import io.github.nearage.jnn.processing.Model;

/**
 * Neural network sample
 *
 * @author Nearage <https://github.com/Nearage>
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
