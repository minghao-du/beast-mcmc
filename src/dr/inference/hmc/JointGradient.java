/*
 * JointGradient.java
 *
 * Copyright © 2002-2024 the BEAST Development Team
 * http://beast.community/about
 *
 * This file is part of BEAST.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * BEAST is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 *  BEAST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAST; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 *
 */

package dr.inference.hmc;

import dr.inference.model.*;
import dr.xml.Reportable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Future;

/**
 * A composite implementation of {@link GradientWrtParameterProvider} that aggregates derivatives
 * from multiple underlying providers.
 * <p>
 * <b>Core Logic:</b>
 * In Bayesian inference, the total log-posterior is often the sum of multiple independent log-likelihood components
 * (e.g., Tree Likelihood + Prior A + Prior B). By the linearity of differentiation, the gradient of the sum
 * is the sum of the gradients:
 * <pre>
 * &nabla; L<sub>total</sub> = &Sigma; &nabla; L<sub>i</sub>
 * </pre>
 * This class acts as a <b>Summation Aggregator</b>. It delegates the calculation to a list of child providers
 * and sums their resulting vectors (or matrices) to produce the total gradient (or Hessian) with respect to
 * a single shared {@link Parameter}.
 * <p>
 * <b>Key Features:</b>
 * <ul>
 * <li><b>Consistency Checking:</b> Ensures all child providers target the exact same {@link Parameter} object
 * with identical dimensions and current values.</li>
 * <li><b>Parallel Execution:</b> Capable of distributing the gradient calculation of child components across
 * multiple threads using a {@link ParallelGradientExecutor}, followed by a reduction (summation) step.</li>
 * <li><b>Hessian Support:</b> Also aggregates second-order derivatives (Hessians) if the underlying providers
 * support it.</li>
 * <li><b>Validation:</b> Implements {@link Reportable} to expose the aggregated gradient to the automatic
 * numerical verification mechanism.</li>
 * </ul>
 *
 * @author Max Tolkoff
 * @author Marc A. Suchard
 */
public class JointGradient implements GradientWrtParameterProvider, HessianWrtParameterProvider,
        DerivativeWrtParameterProvider, Reportable {

    private final int dimension;
    private final Likelihood likelihood;
    private final Parameter parameter;
    private final ParallelGradientExecutor parallelExecutor;

    final List<GradientWrtParameterProvider> derivativeList;

    final List<DerivativeWrtParameterProvider> newDerivativeList;
    private final DerivativeOrder highestOrder;

    /**
     * Constructs a JointGradient aggregator for serial execution.
     * @param derivativeList The list of providers whose gradients should be summed.
     */
    public JointGradient(List<GradientWrtParameterProvider> derivativeList) {
        this(derivativeList, 0);
    }

    /**
     * Constructs a JointGradient aggregator with optional parallel execution.
     * <p>
     * This constructor validates that all providers in {@code derivativeList} correspond to the same
     * parameter dimension and value. It also constructs a {@link CompoundLikelihood} representing
     * the sum of the potentials from all providers.
     *
     * @param derivativeList The list of providers whose gradients should be summed.
     * @param threadCount    The number of threads to use for parallel calculation.
     * If > 1, a {@link ParallelGradientExecutor} is initialized.
     * @throws RuntimeException if the providers target parameters with unequal dimensions or values.
     */
    public JointGradient(List<GradientWrtParameterProvider> derivativeList, int threadCount) {

        this.derivativeList = derivativeList;

        GradientWrtParameterProvider first = derivativeList.get(0);
        dimension = first.getDimension();
        parameter = first.getParameter();

        if (derivativeList.size() == 1) {
            likelihood = first.getLikelihood();
        } else {
            List<Likelihood> likelihoodList = new ArrayList<>();

            for (GradientWrtParameterProvider grad : derivativeList) {
                if (grad.getDimension() != dimension) {
                    throw new RuntimeException("Unequal parameter dimensions");
                }
                if (!Arrays.equals(grad.getParameter().getParameterValues(), parameter.getParameterValues())){
                    throw new RuntimeException("Unequal parameter values");
                }
                Likelihood outer = grad.getLikelihood();
                if (outer instanceof ReciprocalLikelihood) {
                    if (!(likelihoodList.contains(outer))) {
                        likelihoodList.add(outer);
                    }
                } else {
                    for (Likelihood likelihood : grad.getLikelihood().getLikelihoodSet()) {
                        if (!(likelihoodList.contains(likelihood))) {
                            likelihoodList.add(likelihood);
                        }
                    }
                }
            }
            likelihood = new CompoundLikelihood(likelihoodList);
        }

        // NEW

        this.newDerivativeList = new ArrayList<>();
        for (GradientWrtParameterProvider p : derivativeList) {
            if (p instanceof DerivativeWrtParameterProvider) { // TODO Remove if when conversion finished
                DerivativeWrtParameterProvider provider = (DerivativeWrtParameterProvider) p;
                newDerivativeList.add(provider);
            }
        }
        this.highestOrder = DerivativeWrtParameterProvider.getHighestOrder(newDerivativeList);

        // Parallel threading

        if (threadCount > 1 || threadCount < 0) {
            parallelExecutor = new ParallelGradientExecutor(threadCount, derivativeList);
        } else {
            parallelExecutor = null;
        }
    }

    /**
     * returns the aggregate Likelihood (Potential Energy).
     * <p>
     * If multiple providers are involved, this returns a {@link CompoundLikelihood} wrapping
     * all unique likelihood components found in the derivative list.
     */
    @Override
    public Likelihood getLikelihood() {
        return likelihood;
    }

    @Override
    public Parameter getParameter() {
        return parameter;
    }

    @Override
    public int getDimension(DerivativeOrder order) {
        return order.getDerivativeDimension(dimension);
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    /**
     * Computes the total gradient by summing the gradients from all child providers.
     * <p>
     * This method delegates to {@link #getDerivativeLogDensity(DerivativeType)} using
     * {@link DerivativeType#GRADIENT}, which handles either serial iteration or parallel
     * map-reduce depending on the configuration.
     *
     * @return The element-wise sum of all gradient vectors.
     */
    @Override
    public double[] getDerivativeLogDensity(DerivativeOrder type) {

        assert (highestOrder.getValue() >= type.getValue());

        int size = newDerivativeList.size();

        final double[] derivative = newDerivativeList.get(0).getDerivativeLogDensity(type);

        for (int i = 1; i < size; i++) {

            final double[] temp = newDerivativeList.get(i).getDerivativeLogDensity(type);

            for (int j = 0; j < temp.length; j++) {
                derivative[j] += temp[j];
            }
        }

        return derivative;
    }

    @Override
    public DerivativeOrder getHighestOrder() {
        return highestOrder;
    }

    @Override
    public double[] getDiagonalHessianLogDensity() {
        return getDerivativeLogDensity(DerivativeType.DIAGONAL_HESSIAN);
    }

    /**
     * Computes the total Hessian (matrix of second partial derivatives) by summing
     * the Hessians from all child providers.
     * <p>
     * <b>Note:</b> This operation is computationally expensive (O(N<sup>2</sup>)).
     * It asserts that all child providers implement {@link HessianWrtParameterProvider}.
     *
     * @return The element-wise sum of all Hessian matrices.
     */
    @Override
    public double[][] getHessianLogDensity() {
        assert (derivativeList.get(0) instanceof HessianWrtParameterProvider);
        int size = derivativeList.size();

        final double[][] hessian = ((HessianWrtParameterProvider) derivativeList.get(0)).getHessianLogDensity();

        if (DEBUG) {
            // stop timer
            String name = derivativeList.get(0).getLikelihood().getId();
            System.err.println(name);
            System.err.println(Arrays.deepToString(hessian));
        }

        for (int i = 1; i < size; i++) {
            assert (derivativeList.get(i) instanceof HessianWrtParameterProvider);

            final double[][] temp = ((HessianWrtParameterProvider) derivativeList.get(i)).getHessianLogDensity();

            if (DEBUG) {
                String name = derivativeList.get(i).getLikelihood().getId();
                System.err.println(name);
                System.err.println(Arrays.deepToString(temp));
            }

            for (int j = 0; j < temp[0].length; j++) {
                for (int k = 0; k < temp[0].length; k++) {
                    hessian[j][k] += temp[j][k];
                }
            }
        }

        if (DEBUG) {
            if (DEBUG_KILL) {
                System.exit(-1);
            }
        }

        return hessian;
    }

    double[] getDerivativeLogDensity(DerivativeType derivativeType) {
        if (parallelExecutor != null) {
            return getDerivativeLogDensityParallelImpl(derivativeType);
        } else {
            return getDerivativeLogDensitySerialImpl(derivativeType);
        }
    }

    private double[] getDerivativeLogDensityParallelImpl(DerivativeType derivativeType) {

        return parallelExecutor.getDerivativeLogDensityInParallel(derivativeType, (gradients, length) -> {
            double[] reduction = new double[length];
            for (Future<double[]> result : gradients) {
                double[] tmp = result.get();
                for (int j = 0; j < length; ++j) {
                    reduction[j] += tmp[j];
                }
            }
            return reduction;
        }, dimension);
    }

    private double[] getDerivativeLogDensitySerialImpl(DerivativeType derivativeType) {

        int size = derivativeList.size();

        final double[] derivative = derivativeType.getDerivativeLogDensity(derivativeList.get(0));

        for (int i = 1; i < size; i++) {

            final double[] temp = derivativeType.getDerivativeLogDensity(derivativeList.get(i));

            for (int j = 0; j < temp.length; j++) {
                derivative[j] += temp[j];
            }
        }

        return derivative;
    }

    @Override
    public double[] getGradientLogDensity() {
        return getDerivativeLogDensity(DerivativeType.GRADIENT);
    }

    private static final boolean DEBUG = false;
    private static final boolean DEBUG_KILL = false;

    /**
     * Generates a validation report for the aggregated gradient.
     * <p>
     * This triggers the {@link GradientWrtParameterProvider#getReportAndCheckForError} routine,
     * which compares the analytically aggregated gradient against a numerically computed gradient
     * (via finite differences on the total likelihood) to ensure the summation logic and
     * individual gradients are correct.
     *
     * @return A formatted string report comparing analytic vs. numeric gradients.
     * @throws RuntimeException if the difference exceeds the tolerance.
     */
    @Override
    public String getReport() {
        return  "jointGradient." + parameter.getParameterName() + "\n" +
                GradientWrtParameterProvider.getReportAndCheckForError(this,
                Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
                GradientWrtParameterProvider.TOLERANCE);
    }

    enum DerivativeType {
        GRADIENT("gradient") {
            @Override
            public double[] getDerivativeLogDensity(GradientWrtParameterProvider gradientWrtParameterProvider) {
                return gradientWrtParameterProvider.getGradientLogDensity();
            }
        },
        DIAGONAL_HESSIAN("diagonalHessian") {
            @Override
            public double[] getDerivativeLogDensity(GradientWrtParameterProvider gradientWrtParameterProvider) {
                return ((HessianWrtParameterProvider) gradientWrtParameterProvider).getDiagonalHessianLogDensity();
            }
        };

        @SuppressWarnings("unused")
        final private String type;

        DerivativeType(String type) {
            this.type = type;
        }

        public abstract double[] getDerivativeLogDensity(GradientWrtParameterProvider gradientWrtParameterProvider);
    }
}
