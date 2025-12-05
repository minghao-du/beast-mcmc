/*
 * ProcessSimulationDelegate.java
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

package dr.evomodel.treedatalikelihood.preorder;

import dr.evolution.tree.*;
import dr.evomodel.continuous.MultivariateDiffusionModel;
import dr.evomodel.treedatalikelihood.ProcessOnTreeDelegate;
import dr.evomodel.treedatalikelihood.ProcessSimulation;
import dr.evomodel.treedatalikelihood.TreeTraversal;
import dr.evomodel.treedatalikelihood.continuous.*;
import dr.evomodel.treedatalikelihood.continuous.cdi.ContinuousDiffusionIntegrator;
import dr.inference.model.Model;
import dr.inference.model.ModelListener;
import dr.math.matrixAlgebra.*;
import dr.math.matrixAlgebra.CholeskyDecomposition;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;

import java.util.List;
import java.util.Map;

import static dr.math.matrixAlgebra.missingData.MissingOps.*;

/**
 * ProcessSimulationDelegate defines the contract for plugin delegates capable of
 * simulating evolutionary processes along a phylogenetic tree.
 * <p>
 * <b>Core Responsibilities:</b>
 * <ul>
 * <li><b>Stochastic Simulation:</b> Executes forward simulation of traits (e.g., continuous diffusion,
 * discrete substitutions) from the root to the tips of the tree.</li>
 * <li><b>High-Performance Computing:</b> Utilizes vectorized integer arrays (via {@link #vectorizeNodeOperations})
 * instead of object iteration during the core simulation loop to maximize performance and cache locality.</li>
 * <li><b>Data Provision:</b> Extends {@link TreeTraitProvider} to expose the results of the simulation
 * (the simulated traits) to other components like Loggers or Likelihood functions.</li>
 * <li><b>Model Reactivity:</b> Extends {@link ModelListener} to automatically update internal caches
 * (e.g., precision matrices) when underlying model parameters change.</li>
 * </ul>
 * <p>
 * <b>Architecture Note:</b>
 * This interface extends {@link ProcessOnTreeDelegate}, adding the specific capability to <i>write</i>
 * data onto the tree, whereas the parent interface primarily defines tree traversal capabilities.
 *
 * @see ProcessOnTreeDelegate
 * @see TreeTraitProvider
 * @see dr.evomodel.treedatalikelihood.preorder.ProcessSimulationDelegate.AbstractDelegate
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 */
public interface ProcessSimulationDelegate extends ProcessOnTreeDelegate, TreeTraitProvider, ModelListener {

    /**
     * Executes the stochastic simulation process across the tree.
     * <p>
     * This is the core computational method. It iterates through the provided {@code operations} array,
     * applying the evolutionary process (e.g., Brownian motion updates) to each node in the specified order.
     * <p>
     * <b>Performance Note:</b> The operations are passed as a flat primitive {@code int[]} array
     * to avoid object overhead during the tight simulation loop. The structure of this array is determined
     * by {@link #vectorizeNodeOperations}.
     *
     * @param operations     A flattened integer array containing the sequence of simulation instructions.
     * Each logical operation occupies {@link #getSingleOperationSize()} consecutive indices.
     * @param operationCount The total number of logical operations (nodes) to process.
     * @param rootNodeNumber The index of the root node where the simulation begins.
     */
    void simulate(int[] operations, int operationCount, int rootNodeNumber);

    /**
     * Registers the controlling simulation process with this delegate.
     * <p>
     * This allows the delegate to communicate back to the main driver or access global context
     * if necessary. This is typically used for dependency injection after instantiation.
     *
     * @param simulationProcess The main {@link ProcessSimulation} instance driving this delegate.
     */
    void setCallback(ProcessSimulation simulationProcess);

    /**
     * Converts a high-level list of node operations into a flattened, vectorized integer array.
     * <p>
     * This method is called during the setup phase (before the MCMC loop). It translates
     * object-oriented {@link NodeOperations} into the primitive format required by {@link #simulate},
     * optimizing the data structure for repeated execution.
     *
     * @param nodeOperations The list of high-level operation objects describing the tree traversal order.
     * @param operations     The destination array where the vectorized instructions will be written.
     * The array must be pre-allocated with sufficient size.
     * @return The number of logical operations successfully vectorized (typically equal to the list size).
     */
    int vectorizeNodeOperations(List<ProcessOnTreeDelegate.NodeOperation> nodeOperations, int[] operations);

    /**
     * Returns the size (stride) of a single operation tuple in the vectorized array.
     * <p>
     * Since {@code operations} is a 1D array representing a list of tuples, this method defines
     * how many integers constitute one logical step.
     * <p>
     * Example: If an operation requires [nodeIndex, parentIndex, matrixIndex], this method returns 3.
     * This allows the {@link #simulate} loop to increment its pointer correctly.
     *
     * @return The number of integers used to represent a single simulation step.
     */
    int getSingleOperationSize();

    /**
     * Abstract base class for all process simulation delegates on a phylogenetic tree.
     * <p>
     * This class provides a skeletal implementation of the {@link ProcessSimulationDelegate} interface,
     * handling the administrative tasks of tree management, traversal order enforcement, and
     * trait registration.
     * <p>
     * <b>Key Responsibilities:</b>
     * <ul>
     * <li><b>Workflow Management:</b> Implements the {@link #simulate(int[], int, int)} method as a
     * Template Method, defining the strict order of operations (setup -> root -> internal nodes).</li>
     * <li><b>Tree Abstraction:</b> Unwraps potential {@link TransformableTree} wrappers to access and
     * store the underlying base tree.</li>
     * <li><b>Trait Management:</b> Acts as a {@link TreeTraitProvider} by maintaining a helper registry
     * for simulated traits.</li>
     * </ul>
     * <p>
     * <b>Contract for Subclasses:</b>
     * Concrete implementations must define the specific mathematical logic for the simulation by implementing:
     * <ul>
     * <li>{@link #constructTraits(Helper)}: To register output traits during initialization.</li>
     * <li>{@link #setupStatistics()}: To prepare caches or matrices before simulation.</li>
     * <li>{@link #simulateRoot(int)}: To generate the state at the root.</li>
     * <li>{@link #simulateNode(int, int, int, int, int)}: To propagate the state from parent to child.</li>
     * </ul>
     */
    abstract class AbstractDelegate implements ProcessSimulationDelegate {

        /**
         * Constructs a new simulation delegate.
         * <p>
         * <b>Warning:</b> This constructor invokes the abstract method {@link #constructTraits(Helper)}.
         * Subclasses must ensure that their implementation of {@code constructTraits} does not rely on
         * fields that are initialized in the subclass's own constructor, as they will not yet be set
         * when this method is called.
         *
         * @param name The identifier/name for this delegate.
         * @param tree The phylogenetic tree upon which the process runs.
         */
        AbstractDelegate(String name, Tree tree) {
            this.name = name;
            this.tree = tree;
            this.baseTree = getBaseTree(tree);
            constructTraits(treeTraitHelper);
        }

        /**
         * <b>[Hook Method]</b> Constructs and registers the tree traits generated by this simulation.
         * <p>
         * This method is called immediately during the superclass construction.
         * Implementations must instantiate specific {@link TreeTrait} objects and register them
         * using {@code treeTraitHelper.addTrait(trait)}. This ensures traits are available via
         * {@link #getTreeTraits()} as soon as the object is instantiated.
         *
         * @param treeTraitHelper The registry helper to which traits must be added.
         */
        protected abstract void constructTraits(Helper treeTraitHelper);

        /**
         * Returns the optimal traversal type for this simulation.
         * <p>
         * This implementation enforces {@link TreeTraversal.TraversalType#PRE_ORDER} (root-to-tip).
         * Evolutionary simulations are causal processes; the state of a parent node must be resolved
         * before the state of its children can be simulated.
         *
         * @return Always returns {@code PRE_ORDER}.
         */
        @Override
        public final TreeTraversal.TraversalType getOptimalTraversalType() {
            return TreeTraversal.TraversalType.PRE_ORDER;
        }

        /**
         * Registers the main simulation process controller with this delegate.
         *
         * @param simulationProcess The calling process simulation object, used for callbacks.
         */
        @Override
        public final void setCallback(ProcessSimulation simulationProcess) {
            this.simulationProcess = simulationProcess;
        }

        /**
         * <b>[Template Method]</b> Executes the full simulation over the tree.
         * <p>
         * This method orchestrates the simulation lifecycle in the following strict order:
         * <ol>
         * <li>Calls {@link #setupStatistics()} to prepare numerical caches (e.g., matrix decompositions).</li>
         * <li>Calls {@link #simulateRoot(int)} to initialize the root state.</li>
         * <li>Iterates through the {@code operations} array, delegating each step to
         * {@link #simulateNode(int, int, int, int, int)}.</li>
         * </ol>
         *
         * @param operations     A vectorized array of integers containing tree traversal instructions.
         * Each operation consists of a tuple of integers (size defined by the implementation).
         * @param operationCount The total number of operations to perform.
         * @param rootNodeNumber The index of the root node in the tree.
         */
        @Override
        public void simulate(final int[] operations, final int operationCount,
                             final int rootNodeNumber) {

            setupStatistics();

            simulateRoot(rootNodeNumber);

            int k = 0;
            for (int i = 0; i < operationCount; ++i) {
                simulateNode(
                        operations[k    ],
                        operations[k + 1],
                        operations[k + 2],
                        operations[k + 3],
                        operations[k + 4]
                );

                k += ContinuousDiffusionIntegrator.OPERATION_TUPLE_SIZE;
            }
        }

        /**
         * Recursively unwraps {@link TransformableTree} instances to retrieve the underlying original tree.
         * This ensures node indexing is consistent with the simulation data structures.
         *
         * @param derived The potentially wrapped tree.
         * @return The base tree object.
         */
        private static Tree getBaseTree(Tree derived) {
            while (derived instanceof TransformableTree) {
                derived = ((TransformableTree) derived).getOriginalTree();
            }
            return derived;
        }

        static NodeRef getBaseNode(Tree derived, NodeRef node) {
            while (derived instanceof TransformableTree) {
                derived = ((TransformableTree) derived).getOriginalTree();
                node = ((TransformableTree) derived).getOriginalNode(node);
            }
            return node;
        }

        protected double getNormalization() {
            return 1.0;
        }

        @Override
        public final TreeTrait[] getTreeTraits() {
            return treeTraitHelper.getTreeTraits();
        }

        @Override
        public final TreeTrait getTreeTrait(String key) {
            return treeTraitHelper.getTreeTrait(key);
        }

        /**
         * <b>[Abstract Method]</b> Prepares statistical caches before simulation begins.
         * <p>
         * Called once at the beginning of {@link #simulate}. Implementations should use this to
         * update variance matrices, perform decompositions (e.g., Cholesky), or handle any model
         * parameters that may have changed since the last run.
         */
        protected abstract void setupStatistics();

        /**
         * <b>[Abstract Method]</b> Simulates the state at the root node.
         * <p>
         * Implementations should draw the root state from the stationary distribution or
         * a specific root prior.
         *
         * @param rootNumber The index of the root node.
         */
        protected abstract void simulateRoot(final int rootNumber);

        /**
         * <b>[Abstract Method]</b> Simulates the state for a single node (and its incident branch).
         * <p>
         * This method performs the core Markov transition. Given the state of the parent node
         * (implied by the traversal order), it samples the state of the child node.
         * <p>
         * The parameters {@code v0} through {@code v4} are vectorized operation codes derived from
         * the {@code operations} array. Their specific meaning (e.g., node index, parent index,
         * matrix buffer index) depends on the implementation of {@link #vectorizeNodeOperations}.
         *
         * @param v0 Operation parameter 0.
         * @param v1 Operation parameter 1.
         * @param v2 Operation parameter 2.
         * @param v3 Operation parameter 3.
         * @param v4 Operation parameter 4.
         */
        protected abstract void simulateNode(final int v0,
                                             final int v1,
                                             final int v2,
                                             final int v3,
                                             final int v4);

        final TreeTraitProvider.Helper treeTraitHelper = new Helper();

        ProcessSimulation simulationProcess = null;

        final Tree tree;
        final Tree baseTree;
        final String name;
    }

    abstract class AbstractContinuousTraitDelegate extends AbstractDelegate {

        final int dimTrait;
        final int numTraits;
        final int dimNode;

        final MultivariateDiffusionModel diffusionModel;
        final ContinuousTraitPartialsProvider dataModel;
        final ConjugateRootTraitPrior rootPrior;
        final RootProcessDelegate rootProcessDelegate;
        final ContinuousDataLikelihoodDelegate likelihoodDelegate;

        double[] diffusionVariance;
        DenseMatrix64F Vd;
        DenseMatrix64F Pd;

        double[][] cholesky;
        Map<PartiallyMissingInformation.HashedIntArray,
                ConditionalVarianceAndTransform> conditionalMap;

        AbstractContinuousTraitDelegate(String name,
                                        Tree tree,
                                        MultivariateDiffusionModel diffusionModel,
                                        ContinuousTraitPartialsProvider dataModel,
                                        ConjugateRootTraitPrior rootPrior,
                                        ContinuousRateTransformation rateTransformation,
                                        ContinuousDataLikelihoodDelegate likelihoodDelegate) {
            super(name, tree);

            dimTrait = likelihoodDelegate.getTraitDim();
            numTraits = likelihoodDelegate.getTraitCount();
            dimNode = dimTrait * numTraits;
            this.diffusionModel = diffusionModel;
            this.dataModel = dataModel;
            this.rateTransformation = rateTransformation;
            this.rootPrior = rootPrior;
            this.rootProcessDelegate = likelihoodDelegate.getRootProcessDelegate();
            this.likelihoodDelegate = likelihoodDelegate;

            diffusionModel.addModelListener(this);
        }

        @Override
        public int getSingleOperationSize() {
            return ContinuousDiffusionIntegrator.OPERATION_TUPLE_SIZE;
        }

        @Override
        protected final double getNormalization() {
            return rateTransformation.getNormalization();
        }

        final private ContinuousRateTransformation rateTransformation;

        protected boolean isLoggable() {
            return true;
        }

        @Override
        public void modelChangedEvent(Model model, Object object, int index) {
            if (model == diffusionModel) {
                clearCache();
            } else {
                throw new IllegalArgumentException("Unknown model");
            }
        }

        @Override
        public void modelRestored(Model model) {

        }

        @Override
        protected void setupStatistics() {
            if (diffusionVariance == null) {
                double[][] diffusionPrecision = diffusionModel.getPrecisionmatrix();
                diffusionVariance = getVectorizedVarianceFromPrecision(diffusionPrecision);
                Vd = wrap(diffusionVariance, 0, dimTrait, dimTrait);
                Pd = new DenseMatrix64F(diffusionPrecision);
            }
            if (cholesky == null) {
                cholesky = getCholeskyOfVariance(diffusionVariance, dimTrait);
            }
        }

        void clearCache() {
            diffusionVariance = null;
            Vd = null;
            Pd = null;
            cholesky = null;
            conditionalMap = null;
        }

        static double[][] getCholeskyOfVariance(Matrix variance) {
            final double[][] cholesky;
            try {
                cholesky = new CholeskyDecomposition(variance).getL();
            } catch (IllegalDimension illegalDimension) {
                throw new RuntimeException("Attempted Cholesky decomposition on non-square matrix");
            }
            return cholesky;
        }

        static double[][] getCholeskyOfVariance(double[] variance, final int dim) {
            return CholeskyDecomposition.execute(variance, 0, dim);
        }

        static DenseMatrix64F getCholeskyOfVariance(DenseMatrix64F variance, final int dim) {

            org.ejml.interfaces.decomposition.CholeskyDecomposition<DenseMatrix64F> engine =
                    DecompositionFactory.chol(dim, true);
            engine.decompose(variance);

            return engine.getT(null);
        }

//        static WrappedMatrix getCholeskyOfVariance(final ReadableMatrix variance, final int dim) {
//            return CholeskyDecomposition.execute(variance, dim);
//        }

        private static double[] getVectorizedVarianceFromPrecision(double[][] precision) {
            return new SymmetricMatrix(precision).inverse().toArrayComponents();
        }

    }
}
