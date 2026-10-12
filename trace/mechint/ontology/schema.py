"""
MechInt Ontology Schema support: Task provenance, ILP learning, architectural transfer, theory building
Foundation layer 0
"""

from rdflib import Namespace, URIRef
from typing import Dict, Set, Any

MECHINT_BASE = "http://purl.org/mechint"
MECHINT = Namespace(f"{MECHINT_BASE}#")


class MechIntSchema:
    """Comprehensive mechanistic interpretability ontology"""

    def __init__(self):
        self.namespace = MECHINT
        self.base_uri = MECHINT_BASE

    # ========================================
    # CLASSES
    # ========================================
    @property
    def classes(self) -> Dict[str, URIRef]:
        return {
            # ============ LAYER 0: PROVENANCE ============
            'Experiment': self.namespace.Experiment,
            'Task': self.namespace.Task,
            'TaskFamily': self.namespace.TaskFamily,
            'Dataset': self.namespace.Dataset,
            'Intervention': self.namespace.Intervention,
            'Model': self.namespace.Model,
            'Checkpoint': self.namespace.Checkpoint,
            # They represent KNOWN experimental labels, not LEARNED patterns
            # Task-specific circuit types (known from experimental design)
            'TaskSpecificCircuit': self.namespace.TaskSpecificCircuit,

            # Semantic role circuits (provenance labels)
            'RoleCircuit': self.namespace.RoleCircuit,
            'InstrumentCircuit': self.namespace.InstrumentCircuit,
            'LocationCircuit': self.namespace.LocationCircuit,
            'BeneficiaryCircuit': self.namespace.BeneficiaryCircuit,
            'AgentCircuit': self.namespace.AgentCircuit,
            'PatientCircuit': self.namespace.PatientCircuit,
            'ThemeCircuit': self.namespace.ThemeCircuit,
            'TimeCircuit': self.namespace.TimeCircuit,
            'MannerCircuit': self.namespace.MannerCircuit,
            'PurposeCircuit': self.namespace.PurposeCircuit,

            # IOI circuits (provenance labels)
            'IOICircuit': self.namespace.IOICircuit,
            'NameMoverCircuit': self.namespace.NameMoverCircuit,
            'DuplicateTokenCircuit': self.namespace.DuplicateTokenCircuit,
            'SInhibitionCircuit': self.namespace.SInhibitionCircuit,

            # Numerical circuits (provenance labels)
            'NumericalCircuit': self.namespace.NumericalCircuit,
            'MagnitudeComparisonCircuit': self.namespace.MagnitudeComparisonCircuit,
            'DigitExtractionCircuit': self.namespace.DigitExtractionCircuit,

            # ============ LAYER 1: STRUCTURAL ============
            # Base circuit structure
            'Circuit': self.namespace.Circuit,
            'Component': self.namespace.Component,
            'Node': self.namespace.Node,
            'Edge': self.namespace.Edge,

            # Specific components - strictly for Transformer models
            'AttentionHead': self.namespace.AttentionHead,
            'MLPBlock': self.namespace.MLPBlock,
            'LayerNorm': self.namespace.LayerNorm,
            'Residual': self.namespace.Residual,
            'Input': self.namespace.Input,
            'Output': self.namespace.Output,

            # ============ LAYER 2: SEMANTIC ============
            # These classes represent LEARNED knowledge from ILP -
            # Computational patterns (learned/inferred by ILP_sem or CFS)
            'ComputationalPattern': self.namespace.ComputationalPattern,
            'BindingPattern': self.namespace.BindingPattern,
            'ComparisonPattern': self.namespace.ComparisonPattern,
            'SelectionPattern': self.namespace.SelectionPattern,
            'InhibitionPattern': self.namespace.InhibitionPattern,
            'CopyPattern': self.namespace.CopyPattern,
            'ExtractionPattern': self.namespace.ExtractionPattern,

            # ILP learned specifications
            'LearnedSpecification': self.namespace.LearnedSpecification,
            'SemanticSpec': self.namespace.SemanticSpec,  # From ILP_sem
            'ArchitecturalSpec': self.namespace.ArchitecturalSpec,  # From ILP_arch

            # Predicate class for structured P_H storage
            'Predicate': self.namespace.Predicate,  # Individual predicate in P_H

            # ============ LAYER 3: ARCHITECTURAL ============
            'ArchitecturalSignature': self.namespace.ArchitecturalSignature,
            'Motif': self.namespace.Motif,
            'ComponentStructure': self.namespace.ComponentStructure,

            # ============ LAYER 4: META-KNOWLEDGE ============
            'Theory': self.namespace.Theory,
            'MetaTheory': self.namespace.MetaTheory,
            'CircuitFamily': self.namespace.CircuitFamily,
            'Subsumption': self.namespace.Subsumption,
            'Equivalence': self.namespace.Equivalence,
        }

    # ========================================
    # OBJECT PROPERTIES
    # ========================================
    @property
    def object_properties(self) -> Dict[str, URIRef]:
        return {
            # ============ LAYER 0: PROVENANCE ============
            'discoveredIn': self.namespace.discoveredIn,  # Circuit → Experiment
            'testedOn': self.namespace.testedOn,  # Circuit → Task
            'belongsToTaskFamily': self.namespace.belongsToTaskFamily,  # Circuit → TaskFamily
            'usedDataset': self.namespace.usedDataset,  # Experiment → Dataset
            'fromModel': self.namespace.fromModel,  # Circuit → Model
            'atCheckpoint': self.namespace.atCheckpoint,  # Circuit → Checkpoint

            # Provenance label (what we DESIGNED to test)
            'hasExperimentalLabel': self.namespace.hasExperimentalLabel,  # Circuit → TaskSpecificCircuit subclass
            'targetsPhenomenon': self.namespace.targetsPhenomenon,  # Circuit → phenomenon string

            # ============ LAYER 1: STRUCTURAL ============
            'hasComponent': self.namespace.hasComponent,  # Circuit → Component
            'hasMemberNode': self.namespace.hasMemberNode,  # Circuit → Node
            'hasMemberEdge': self.namespace.hasMemberEdge,  # Circuit → Edge
            'connectsTo': self.namespace.connectsTo,  # Component → Component
            'hasSource': self.namespace.hasSource,  # Edge → Node
            'hasTarget': self.namespace.hasTarget,  # Edge → Node

            # ============ LAYER 2: SEMANTIC ============
            'hasSemanticSpec': self.namespace.hasSemanticSpec,  # Circuit → SemanticSpec
            'hasComputationalPattern': self.namespace.hasComputationalPattern,  # Circuit → Pattern
            'implementsFunction': self.namespace.implementsFunction,  # Circuit → function
            'operatesOn': self.namespace.operatesOn,  # Circuit → entities

            # Predicate set (P_H) - structured storage for set operations
            'hasPredicateInSet': self.namespace.hasPredicateInSet,  # SemanticSpec → Predicate (multi-valued)
            # Enables: Jaccard(P1, P2) = |P1 ∩ P2| / |P1 ∪ P2|, θ-subsumption (P_A ⊆ P_B)
            # Usage: Each predicate is separate triple, allowing SPARQL COUNT/FILTER operations

            # Pattern relationships (for RQ2)
            'subsumes': self.namespace.subsumes,  # Pattern A subsumes B
            'equivalentTo': self.namespace.equivalentTo,  # Pattern A ≡ B
            'refinementOf': self.namespace.refinementOf,  # Spec B refines A
            'sharesPredicatesWith': self.namespace.sharesPredicatesWith,

            # ============ LAYER 3: ARCHITECTURAL ============
            'hasArchitecturalSignature': self.namespace.hasArchitecturalSignature,
            'hasMotif': self.namespace.hasMotif,  # Circuit → Motif
            'hasComponentStructure': self.namespace.hasComponentStructure,
            'transfersTo': self.namespace.transfersTo,  # Circuit A → Circuit B (RQ3)
            'sourceCircuit': self.namespace.sourceCircuit,  # Transfer provenance
            'targetCircuit': self.namespace.targetCircuit,

            # ============ LAYER 4: META-KNOWLEDGE ============
            'memberOfFamily': self.namespace.memberOfFamily,  # Circuit → Family
            'supportsTheory': self.namespace.supportsTheory,  # Circuit → Theory
            'contradicts': self.namespace.contradicts,  # Spec A contradicts B
            'generalizationOf': self.namespace.generalizationOf,  # Theory unifies multiple
        }

    # ========================================
    # DATA PROPERTIES
    # ========================================
    @property
    def data_properties(self) -> Dict[str, URIRef]:
        return {
            # ============ LAYER 0: PROVENANCE ============
            'taskName': self.namespace.taskName,
            'roleName': self.namespace.roleName,  # e.g., "INSTRUMENT"
            'phenomenonName': self.namespace.phenomenonName,  # e.g., "NameMover"
            'datasetPath': self.namespace.datasetPath,
            'interventionType': self.namespace.interventionType,
            'modelName': self.namespace.modelName,
            'step': self.namespace.step,
            'timestamp': self.namespace.timestamp,

            # ============ LAYER 1: STRUCTURAL ============
            'inLayer': self.namespace.inLayer,  # int: layer number
            'head': self.namespace.head,  # int: head index
            'address': self.namespace.address,  # string: e.g. "a2.h3"
            'score': self.namespace.score,  # float: attribution
            'rank': self.namespace.rank,  # int: edge rank
            'inGraph': self.namespace.inGraph,  # bool: in circuit?
            'topK': self.namespace.topK,  # int: top-k threshold

            # ============ LAYER 2: SEMANTIC ============
            # Learned specification properties
            'specText': self.namespace.specText,  # FOL clause text (φ as string)
            'confidence': self.namespace.confidence,  # float: ILP confidence score
            'predicates': self.namespace.predicates,  # string: comma-separated (P_H as string)

            # ILP signature S_ILP(φ) = ⟨n_clauses, d̄, P_H, l̄, complexity⟩ (see this in the paper: 2.3)
            # Mapping schema property names to draft notation:
            'numClauses': self.namespace.numClauses,  # int: n_clauses (number of clauses in φ)
            'avgClauseDepth': self.namespace.avgClauseDepth,  # float: d̄ (mean derivation depth)
            'avgClauseLength': self.namespace.avgClauseLength,  # float: l̄ (mean literals per clause)
            'complexity': self.namespace.complexity,  # float: complexity (composite score)
            # Note: P_H (predicate set) stored TWO ways:
            #   1. 'predicates' property (string, backward compatible)
            #   2. hasPredicateInSet → Predicate (structured, for set operations )

            'learnedBy': self.namespace.learnedBy,  # string: "ILP_sem" or "ILP_arch"

            # Predicate class properties (for structured P_H storage)
            'predicateName': self.namespace.predicateName,  # string: predicate name (e.g., "binds")
            'predicateArity': self.namespace.predicateArity,  # int: number of arguments

            'HubComponent': self.namespace.HubComponent,
            'SkipConnection': self.namespace.SkipConnection,

            # In object_properties:
            'hasHubComponent': self.namespace.hasHubComponent,
            'hasSkipConnection': self.namespace.hasSkipConnection,

            # In data_properties:
            'degree': self.namespace.degree,
            'layerGap': self.namespace.layerGap,
            'motifType': self.namespace.motifType,
            'motifPath': self.namespace.motifPath,

            # Evaluation metrics (part of validation statistics κ from the paper 2.4)
            'faithfulness': self.namespace.faithfulness,  # float: normalized faithfulness
            'faithfulnessSimple': self.namespace.faithfulnessSimple,  # float: simple ratio (circuit/baseline)
            'baselinePerformance': self.namespace.baselinePerformance,  # float: Acc_T(M) - full model
            'circuitPerformance': self.namespace.circuitPerformance,  # float: Acc_T(M with C)
            'ablatedPerformance': self.namespace.ablatedPerformance,  # float: Acc_T(M with ∅) - all ablated
            'causalEffect': self.namespace.causalEffect,  # float: Δ_T (paper 2.4) - ablation drop
            'performanceDrop': self.namespace.performanceDrop,  # float: alternative Δ_T calculation
            'performanceDropPct': self.namespace.performanceDropPct,  # float: percentage drop

            # Validation statistics κ = ⟨Acc_T, Δ_T, confidence, ε-minimality⟩ (paper 2.4)
            'taskAccuracy': self.namespace.taskAccuracy,  # float: Acc_T (task accuracy - same as baseline)
            'epsilonMinimal': self.namespace.epsilonMinimal,  # bool: ε-minimality flag
            'permutationPValue': self.namespace.permutationPValue,  # float: permutation test p-value
            # Note: κ = ⟨Acc_T, Δ_T, confidence, ε-minimality⟩ - see Issue #7 for missing components

            # ============ LAYER 3: ARCHITECTURAL ============
            'hasRelativeDepth': self.namespace.hasRelativeDepth,  # float: ρ(ℓ) = ℓ/L (normalized depth, paper 2.3)
            'depthMin': self.namespace.depthMin,  # float: ρ_min (minimum relative depth)
            'depthMax': self.namespace.depthMax,  # float: ρ_max (maximum relative depth)
            'depthMean': self.namespace.depthMean,  # float: ρ̄ (mean relative depth)
            'componentRatio': self.namespace.componentRatio,  # string: "3:1" (attn:mlp ratio)
            'numAttentionHeads': self.namespace.numAttentionHeads,  # int: attention head count
            'numMLPBlocks': self.namespace.numMLPBlocks,  # int: MLP block count
            'signatureText': self.namespace.signatureText,  # string: τ_arch (architectural signature rule)

            # ============ LAYER 4: META-KNOWLEDGE ============
            'jaccardSimilarity': self.namespace.jaccardSimilarity,  # float: predicate overlap
            'predicateOverlap': self.namespace.predicateOverlap,  # int: shared predicates
            'transferSuccess': self.namespace.transferSuccess,  # bool
            'transferPreservation': self.namespace.transferPreservation,  # float: |Δ_src - Δ_tgt|
        }

    # ========================================
    # ENUMERATED INDIVIDUALS
    # ========================================

    @property
    def task_families(self) -> Dict[str, URIRef]:
        """Task family instances"""
        return {
            'SemanticRoles': self.namespace.SemanticRoles,
            'IOI': self.namespace.IOI,
            'NumericalComparison': self.namespace.NumericalComparison,
            'ModularAddition': self.namespace.ModularAddition,
            'GreaterThan': self.namespace.GreaterThan,
        }

    @property
    def motif_types(self) -> Dict[str, URIRef]:
        """Architectural motif instances"""
        return {
            'BackwardAttention': self.namespace.BackwardAttention,
            'ForwardAttention': self.namespace.ForwardAttention,
            'TokenCopy': self.namespace.TokenCopy,
            'PositionalSelection': self.namespace.PositionalSelection,
            'FeedForwardChain': self.namespace.FeedForwardChain,
            'ResidualSkip': self.namespace.ResidualSkip,
            'InductionHead': self.namespace.InductionHead,
        }

    @property
    def computational_patterns(self) -> Dict[str, URIRef]:
        """Computational pattern instances (for classification)"""
        return {
            'Binding': self.namespace.Binding,
            'Comparison': self.namespace.Comparison,
            'Selection': self.namespace.Selection,
            'Inhibition': self.namespace.Inhibition,
            'Copy': self.namespace.Copy,
            'Extraction': self.namespace.Extraction,
        }

    # ========================================
    # PROPERTY SIGNATURES (Domain/Range)
    # ========================================

    def get_property_signatures(self) -> Dict[str, Dict[str, str]]:
        """
        Get domain and range constraints for all properties for per-predicate admissible universe construction.

        Returns:
            Dictionary mapping property name → {'domain': class_name, 'range': class_name/datatype}

        Example:
            {
                'hasComponent': {'domain': 'Circuit', 'range': 'Component'},
                'confidence': {'domain': 'LearnedSpecification', 'range': 'xsd:float'}
            }
        """
        return {
            # ========================================
            # LAYER 0: PROVENANCE
            # ========================================
            'discoveredIn': {
                'domain': 'Circuit',
                'range': 'Experiment',
                'description': 'Links circuit to the experiment in which it was discovered'
            },
            'testedOn': {
                'domain': 'Circuit',
                'range': 'Task',
                'description': 'Links circuit to the task it was tested on'
            },
            'belongsToTaskFamily': {
                'domain': 'Circuit',
                'range': 'TaskFamily',
                'description': 'Links circuit to its task family (e.g., SemanticRoles, IOI)'
            },
            'usedDataset': {
                'domain': 'Experiment',
                'range': 'Dataset',
                'description': 'Links experiment to the dataset used'
            },
            'fromModel': {
                'domain': 'Circuit',
                'range': 'Model',
                'description': 'Links circuit to the model from which it was extracted'
            },
            'atCheckpoint': {
                'domain': 'Circuit',
                'range': 'Checkpoint',
                'description': 'Links circuit to specific model checkpoint'
            },
            'hasExperimentalLabel': {
                'domain': 'Circuit',
                'range': 'TaskSpecificCircuit',
                'description': 'Links circuit to its provenance label (what we designed to test)'
            },
            'targetsPhenomenon': {
                'domain': 'Circuit',
                'range': 'xsd:string',
                'description': 'Phenomenon name this circuit was designed to test'
            },

            # ========================================
            # LAYER 1: STRUCTURAL
            # ========================================
            'hasComponent': {
                'domain': 'Circuit',
                'range': 'Component',
                'description': 'Links circuit to a component (node) it contains'
            },
            'hasMemberNode': {
                'domain': 'Circuit',
                'range': 'Node',
                'description': 'Links circuit to a member node'
            },
            'hasMemberEdge': {
                'domain': 'Circuit',
                'range': 'Edge',
                'description': 'Links circuit to a member edge'
            },
            'connectsTo': {
                'domain': 'Component',
                'range': 'Component',
                'description': 'Links one component to another it connects to'
            },
            'hasSource': {
                'domain': 'Edge',
                'range': 'Node',
                'description': 'Links edge to its source node'
            },
            'hasTarget': {
                'domain': 'Edge',
                'range': 'Node',
                'description': 'Links edge to its target node'
            },

            # ========================================
            # LAYER 2: SEMANTIC
            # ========================================
            'hasSemanticSpec': {
                'domain': 'Circuit',
                'range': 'SemanticSpec',
                'description': 'Links circuit to its learned semantic specification (φ)'
            },
            'hasComputationalPattern': {
                'domain': 'Circuit',
                'range': 'ComputationalPattern',
                'description': 'Links circuit to its inferred computational pattern (learned)'
            },
            'hasPredicateInSet': {
                'domain': 'SemanticSpec',
                'range': 'Predicate',
                'description': 'Links specification to a predicate in its predicate set P_H (multi-valued for set operations)'
            },
            'implementsFunction': {
                'domain': 'Circuit',
                'range': 'xsd:string',
                'description': 'Function description the circuit implements'
            },
            'operatesOn': {
                'domain': 'Circuit',
                'range': 'xsd:string',
                'description': 'Entities the circuit operates on'
            },
            'subsumes': {
                'domain': 'ComputationalPattern',
                'range': 'ComputationalPattern',
                'description': 'Pattern A subsumes pattern B (θ-subsumption)'
            },
            'equivalentTo': {
                'domain': 'ComputationalPattern',
                'range': 'ComputationalPattern',
                'description': 'Pattern A is equivalent to pattern B'
            },
            'refinementOf': {
                'domain': 'SemanticSpec',
                'range': 'SemanticSpec',
                'description': 'Specification B is a refinement of specification A'
            },
            'sharesPredicatesWith': {
                'domain': 'SemanticSpec',
                'range': 'SemanticSpec',
                'description': 'Two specifications share predicates'
            },

            # ========================================
            # LAYER 3: ARCHITECTURAL
            # ========================================
            'hasArchitecturalSignature': {
                'domain': 'Circuit',
                'range': 'ArchitecturalSignature',
                'description': 'Links circuit to its architectural signature (τ_arch)'
            },
            'hasMotif': {
                'domain': 'Circuit',
                'range': 'Motif',
                'description': 'Links circuit to a structural motif it contains'
            },
            'hasComponentStructure': {
                'domain': 'Circuit',
                'range': 'ComponentStructure',
                'description': 'Links circuit to its component structure description'
            },
            'transfersTo': {
                'domain': 'Circuit',
                'range': 'Circuit',
                'description': 'Circuit A transfers to circuit B (RQ3)'
            },
            'sourceCircuit': {
                'domain': 'Transfer',
                'range': 'Circuit',
                'description': 'Source circuit in transfer relationship'
            },
            'targetCircuit': {
                'domain': 'Transfer',
                'range': 'Circuit',
                'description': 'Target circuit in transfer relationship'
            },

            # ========================================
            # LAYER 4: META-KNOWLEDGE
            # ========================================
            'memberOfFamily': {
                'domain': 'Circuit',
                'range': 'CircuitFamily',
                'description': 'Circuit is member of a circuit family'
            },
            'supportsTheory': {
                'domain': 'Circuit',
                'range': 'Theory',
                'description': 'Circuit provides evidence for a theory'
            },
            'contradicts': {
                'domain': 'SemanticSpec',
                'range': 'SemanticSpec',
                'description': 'Specification A contradicts specification B'
            },
            'generalizationOf': {
                'domain': 'Theory',
                'range': 'SemanticSpec',
                'description': 'Theory generalizes multiple specifications'
            },
        }

    def get_datatype_signatures(self) -> Dict[str, Dict[str, str]]:
        """
        Get domain and range (datatype) constraints for data properties.

        Returns:
            Dictionary mapping data property name → {'domain': class_name, 'range': xsd_datatype}
        """
        return {
            # ========================================
            # LAYER 0: PROVENANCE
            # ========================================
            'taskName': {'domain': 'Circuit', 'range': 'xsd:string'},
            'roleName': {'domain': 'Circuit', 'range': 'xsd:string'},
            'phenomenonName': {'domain': 'Circuit', 'range': 'xsd:string'},
            'datasetPath': {'domain': 'Dataset', 'range': 'xsd:string'},
            'interventionType': {'domain': 'Intervention', 'range': 'xsd:string'},
            'modelName': {'domain': 'Model', 'range': 'xsd:string'},
            'step': {'domain': 'Checkpoint', 'range': 'xsd:integer'},
            'timestamp': {'domain': 'Circuit', 'range': 'xsd:dateTime'},

            # ========================================
            # LAYER 1: STRUCTURAL
            # ========================================
            'inLayer': {'domain': 'Component', 'range': 'xsd:integer'},
            'head': {'domain': 'AttentionHead', 'range': 'xsd:integer'},
            'address': {'domain': 'Component', 'range': 'xsd:string'},
            'score': {'domain': 'Edge', 'range': 'xsd:float'},
            'rank': {'domain': 'Edge', 'range': 'xsd:integer'},
            'inGraph': {'domain': 'Component', 'range': 'xsd:boolean'},
            'topK': {'domain': 'Circuit', 'range': 'xsd:integer'},

            # Metrics (part of validation statistics κ )
            'faithfulness': {  # Draft notation: part of κ (but computed differently)
                'domain': 'Circuit',
                'range': 'xsd:float',
                'draft_name': 'faithfulness',
                'description': 'Normalized faithfulness: (circuit_perf - ablated_perf) / (baseline_perf - ablated_perf)'
            },
            'faithfulnessSimple': {
                'domain': 'Circuit',
                'range': 'xsd:float',
                'description': 'Simple faithfulness ratio: circuit_perf / baseline_perf'
            },
            'baselinePerformance': {  # Draft notation: Acc_task_T(M) - full model performance
                'domain': 'Circuit',
                'range': 'xsd:float',
                'draft_name': 'baseline_performance',
                'description': 'Full model task accuracy (M with all edges intact)'
            },
            'circuitPerformance': {  # Draft notation: related to Acc_T(C, φ)
                'domain': 'Circuit',
                'range': 'xsd:float',
                'draft_name': 'circuit_performance',
                'description': 'Circuit task accuracy (M with only circuit edges)'
            },
            'ablatedPerformance': {
                'domain': 'Circuit',
                'range': 'xsd:float',
                'description': 'Fully ablated performance (M with no edges - all ablated)'
            },
            'causalEffect': {  # Draft notation: Δ_T(C) - causal relevance (§2.4)
                'domain': 'Circuit',
                'range': 'xsd:float',
                'draft_name': 'Δ_T',
                'description': 'Drop in task accuracy upon ablation: Acc_T(M) - Acc_T(M_ablate,C)'
            },
            'performanceDrop': {
                'domain': 'Circuit',
                'range': 'xsd:float',
                'description': 'Performance drop: baseline - circuit (alternative Δ_T)'
            },
            'performanceDropPct': {
                'domain': 'Circuit',
                'range': 'xsd:float',
                'description': 'Performance drop as percentage'
            },

            # Validation statistics κ = ⟨Acc_T, Δ_T, confidence, ε-minimality⟩ (§2.4)
            'taskAccuracy': {  # Draft notation: Acc_T
                'domain': 'Circuit',
                'range': 'xsd:float',
                'draft_name': 'Acc_T',
                'description': 'Task accuracy (same as baseline_performance, stored for κ completeness)'
            },
            'epsilonMinimal': {  # Draft notation: ε-minimal
                'domain': 'Circuit',
                'range': 'xsd:boolean',
                'draft_name': 'ε-minimal',
                'description': 'Whether circuit is ε-minimal (greedy ablation validated)'
            },
            'permutationPValue': {
                'domain': 'Circuit',
                'range': 'xsd:float',
                'description': 'Permutation test p-value for statistical significance'
            },

            # κ = ⟨Acc_T, Δ_T, confidence, ε-minimality⟩ (paper: 2.4)
            # Now stored: Acc_T (taskAccuracy), Δ_T (causalEffect)
            # Added but not computed: ε-minimality (epsilonMinimal), p-value (permutationPValue)
            # Confidence stored in SemanticSpec (added by ILP_sem)

            # ========================================
            # LAYER 2: SEMANTIC
            # ========================================
            'specText': {'domain': 'LearnedSpecification', 'range': 'xsd:string'},
            'confidence': {'domain': 'LearnedSpecification', 'range': 'xsd:float'},
            'predicates': {'domain': 'LearnedSpecification', 'range': 'xsd:string'},

            # ILP signature components: S_ILP(φ) = ⟨n_clauses, d̄, P_H, l̄, complexity⟩ (paper 2.3)
            'numClauses': {  # Draft notation: n_clauses
                'domain': 'SemanticSpec',
                'range': 'xsd:integer',
                'draft_name': 'n_clauses',
                'description': 'Number of clauses in learned specification φ'
            },
            'avgClauseDepth': {  # Draft notation: d̄ (d-bar)
                'domain': 'SemanticSpec',
                'range': 'xsd:float',
                'draft_name': 'd̄',
                'description': 'Mean derivation depth (max predicate dependencies along proof path)'
            },
            'avgClauseLength': {  # Draft notation: l̄ (l-bar)
                'domain': 'SemanticSpec',
                'range': 'xsd:float',
                'draft_name': 'l̄',
                'description': 'Mean clause length (number of literals per clause)'
            },
            'complexity': {  # Draft notation: complexity (composite score)
                'domain': 'SemanticSpec',
                'range': 'xsd:float',
                'draft_name': 'complexity',
                'description': 'Composite score: clause count + derivation depth + predicate arity'
            },
            # NOTE: P_H (predicate set) stored in 'predicates' property (Issue #6 for structured storage)

            'learnedBy': {'domain': 'LearnedSpecification', 'range': 'xsd:string'},

            # Predicate class properties (for structured P_H )
            'predicateName': {
                'domain': 'Predicate',
                'range': 'xsd:string',
                'description': 'Name of the predicate (e.g., "binds", "scaffold", "compatible")'
            },
            'predicateArity': {
                'domain': 'Predicate',
                'range': 'xsd:integer',
                'description': 'Number of arguments this predicate takes'
            },

            # ========================================
            # LAYER 3: ARCHITECTURAL
            # ========================================
            'hasRelativeDepth': {  # Draft notation: ρ(ℓ) = ℓ/L (normalized depth, §2.3)
                'domain': 'Component',
                'range': 'xsd:float',
                'draft_name': 'ρ(ℓ)',
                'description': 'Relative layer position: ℓ/L ∈ [0,1] where L is total layers'
            },
            'depthMin': {  # Draft notation: ρ_min (minimum relative depth in circuit)
                'domain': 'Circuit',
                'range': 'xsd:float',
                'draft_name': 'ρ_min',
                'description': 'Minimum relative depth across circuit components'
            },
            'depthMax': {  # Draft notation: ρ_max (maximum relative depth in circuit)
                'domain': 'Circuit',
                'range': 'xsd:float',
                'draft_name': 'ρ_max',
                'description': 'Maximum relative depth across circuit components'
            },
            'depthMean': {  # Draft notation: ρ_mean (mean relative depth)
                'domain': 'Circuit',
                'range': 'xsd:float',
                'draft_name': 'ρ̄',
                'description': 'Mean relative depth across circuit components'
            },
            'componentRatio': {  # Draft notation: component ratio (e.g., "3:1" for attn:mlp)
                'domain': 'Circuit',
                'range': 'xsd:string',
                'draft_name': 'component_ratio',
                'description': 'Ratio of attention heads to MLP blocks (e.g., "3:0", "5:2")'
            },
            'numAttentionHeads': {
                'domain': 'Circuit',
                'range': 'xsd:integer',
                'description': 'Count of attention head components in circuit'
            },
            'numMLPBlocks': {
                'domain': 'Circuit',
                'range': 'xsd:integer',
                'description': 'Count of MLP block components in circuit'
            },
            'signatureText': {  # Draft notation: τ_arch (architectural signature)
                'domain': 'ArchitecturalSignature',
                'range': 'xsd:string',
                'draft_name': 'τ_arch',
                'description': 'Architectural signature rule text (learned by ILP_arch)'
            },

            # Motif details
            'degree': {'domain': 'HubComponent', 'range': 'xsd:integer'},
            'layerGap': {'domain': 'SkipConnection', 'range': 'xsd:integer'},
            'motifType': {'domain': 'Motif', 'range': 'xsd:string'},
            'motifPath': {'domain': 'Motif', 'range': 'xsd:string'},

            # ========================================
            # LAYER 4: META-KNOWLEDGE
            # ========================================
            'jaccardSimilarity': {  # Draft notation: Jaccard(P_H1, P_H2) for predicate overlap
                'domain': 'Comparison',
                'range': 'xsd:float',
                'draft_name': 'Jaccard',
                'description': 'Jaccard similarity of predicate sets: |P1 ∩ P2| / |P1 ∪ P2|'
            },
            'predicateOverlap': {
                'domain': 'Comparison',
                'range': 'xsd:integer',
                'description': 'Count of shared predicates: |P1 ∩ P2|'
            },
            'transferSuccess': {
                'domain': 'Transfer',
                'range': 'xsd:boolean',
                'description': 'Whether transfer met validation thresholds (§2.6)'
            },
            'transferPreservation': {  # Draft notation: |Δ_src_T - Δ_tgt_T| (causal preservation)
                'domain': 'Transfer',
                'range': 'xsd:float',
                'draft_name': '|Δ_src - Δ_tgt|',
                'description': 'Absolute difference in causal effects: |Δ_T(C_src) - Δ_T(C_tgt)|'
            },
        }

    # ========================================
    # INFERENCE RULES (OWL Axioms)
    # ========================================

    def get_inference_axioms(self) -> list:
        """
        OWL 2 axioms for automatic classification and inference.

        This method returns ONLY valid OWL 2 axioms (subclass relations).

        SPARQL-based inference rules (pattern detection, subsumption, transfer matching)
        are NOT valid OWL syntax and must be implemented separately in:
        - Query layer (sparql_interface.py): SPARQL CONSTRUCT/SELECT queries
        - Comparison layer: θ-subsumption logic
        - Transfer layer: Architectural signature matching

        OWL 2 Axioms (Returned Here):
        -----------------------------
        These are processed by OWL reasoners and enable:
        - Automatic classification (e.g., InstrumentCircuit → RoleCircuit → TaskSpecificCircuit)
        - Transitive inference (if A ⊑ B and B ⊑ C, then A ⊑ C)
        - Consistency checking with integrity constraints

        Valid OWL 2 Manchester Syntax:
        - Subclass: "InstrumentCircuit ⊑ RoleCircuit"
        - Disjoint: "DisjointClasses(A, B, C)"
        - Functional: "FunctionalObjectProperty(hasSemanticSpec)"

        SPARQL Rules (NOT Returned Here):
        ---------------------------------
        These require custom implementation in application code.

        Example 1: Pattern Inference from Learned Predicates
        -----------------------------------------------------
        RULE (conceptual):
            "hasComputationalPattern some BindingPattern ← predicates contains 'binds'"

        IMPLEMENTATION (in sparql_interface.py or ILP module):
            ```python
            def infer_computational_pattern(circuit_uri, learned_predicates):
                if 'binds' in learned_predicates:
                    graph.add((circuit_uri, MECHINT.hasComputationalPattern,
                              MECHINT.BindingPattern))
                elif 'compares' in learned_predicates:
                    graph.add((circuit_uri, MECHINT.hasComputationalPattern,
                              MECHINT.ComparisonPattern))
                # ... etc
            ```

        Example 2: Transfer Compatibility Matching
        ------------------------------------------
        RULE (conceptual):
            "transfersTo some Circuit ← hasRelativeDepth same, hasMotif same"

        IMPLEMENTATION (in transfer module):
            ```python
            def find_transfer_candidates(source_circuit_uri):
                query = '''
                SELECT ?candidate WHERE {
                    # Get source features
                    BIND(<{source}> as ?src)
                    ?src mechint:depthMean ?srcDepth .
                    ?src mechint:hasMotif ?srcMotif .

                    # Find candidates with similar features
                    ?candidate mechint:depthMean ?candDepth .
                    ?candidate mechint:hasMotif ?candMotif .

                    FILTER(ABS(?candDepth - ?srcDepth) < 0.1)
                    FILTER(?candMotif = ?srcMotif)
                }
                '''
                return execute_query(query)
            ```

        Example 3: θ-Subsumption (Comparison Layer)
        ------------------------------------------
        RULE (conceptual):
            "subsumes(?A, ?B) ← predicates(?A) ⊇ predicates(?B)"

        IMPLEMENTATION (in comparison module):
            ```python
            def check_subsumption(spec_a_uri, spec_b_uri):
                # Get predicate sets
                preds_a = get_predicates(spec_a_uri)  # e.g., {'binds', 'scaffold'}
                preds_b = get_predicates(spec_b_uri)  # e.g., {'binds', 'scaffold', 'compatible'}

                # Check set inclusion
                if preds_a.issubset(preds_b):
                    # A subsumes B (A is more general)
                    graph.add((spec_a_uri, MECHINT.subsumes, spec_b_uri))
                    return True
                return False
            ```

        Returns:
            List of valid OWL 2 axioms in Manchester syntax (subclass relations only)
        """
        return [
            # ========================================
            # LAYER 0: PROVENANCE HIERARCHY
            # ========================================
            # These are KNOWN experimental design labels, not learned patterns

            # Semantic role circuit hierarchy
            "InstrumentCircuit ⊑ RoleCircuit",
            "LocationCircuit ⊑ RoleCircuit",
            "BeneficiaryCircuit ⊑ RoleCircuit",
            "AgentCircuit ⊑ RoleCircuit",
            "PatientCircuit ⊑ RoleCircuit",
            "ThemeCircuit ⊑ RoleCircuit",
            "TimeCircuit ⊑ RoleCircuit",
            "MannerCircuit ⊑ RoleCircuit",
            "PurposeCircuit ⊑ RoleCircuit",
            "RoleCircuit ⊑ TaskSpecificCircuit",

            # IOI circuit hierarchy
            "NameMoverCircuit ⊑ IOICircuit",
            "DuplicateTokenCircuit ⊑ IOICircuit",
            "SInhibitionCircuit ⊑ IOICircuit",
            "IOICircuit ⊑ TaskSpecificCircuit",

            # Numerical circuit hierarchy
            "MagnitudeComparisonCircuit ⊑ NumericalCircuit",
            "DigitExtractionCircuit ⊑ NumericalCircuit",
            "NumericalCircuit ⊑ TaskSpecificCircuit",

            # ========================================
            # LAYER 2: SEMANTIC PATTERN HIERARCHY
            # ========================================
            # These are LEARNED computational patterns from ILP

            # All patterns are computational patterns
            "BindingPattern ⊑ ComputationalPattern",
            "ComparisonPattern ⊑ ComputationalPattern",
            "SelectionPattern ⊑ ComputationalPattern",
            "InhibitionPattern ⊑ ComputationalPattern",
            "CopyPattern ⊑ ComputationalPattern",
            "ExtractionPattern ⊑ ComputationalPattern",
        ]

    def get_integrity_constraints(self) -> list:
        """
        OWL 2 integrity constraints for consistency checking and contradiction detection.
        Required for 2.5 comparison: detecting when φ₁ ∪ φ₂ ∪ B_sem violates constraints.

        These constraints enforce:
        - Mutual exclusivity (disjointness)
        - Single-valued properties (functional)
        - Required relationships (cardinality)
        - Type safety (domain/range)

        Returns:
            List of OWL 2 constraint axioms in Manchester syntax
        """
        return [
            # ========================================
            # DISJOINTNESS CONSTRAINTS
            # ========================================

            # LAYER 2: Computational patterns are mutually exclusive (LEARNED)
            "DisjointClasses(BindingPattern, ComparisonPattern, SelectionPattern, "
            "InhibitionPattern, CopyPattern, ExtractionPattern)",

            # LAYER 0: Specific role circuits are mutually exclusive (PROVENANCE)
            # NOTE: These are provenance labels, not learned patterns
            # A circuit has ONE experimental label but may have MULTIPLE learned patterns
            "DisjointClasses(InstrumentCircuit, LocationCircuit, BeneficiaryCircuit, "
            "AgentCircuit, PatientCircuit, ThemeCircuit, TimeCircuit, MannerCircuit, PurposeCircuit)",

            # LAYER 0: IOI circuits are mutually exclusive (PROVENANCE)
            "DisjointClasses(NameMoverCircuit, DuplicateTokenCircuit, SInhibitionCircuit)",

            # LAYER 0: Numerical circuits are mutually exclusive (PROVENANCE)
            "DisjointClasses(MagnitudeComparisonCircuit, DigitExtractionCircuit)",

            # LAYER 1: Component types are mutually exclusive (STRUCTURAL)
            "DisjointClasses(AttentionHead, MLPBlock, LayerNorm, Residual, Input, Output)",

            # IMPORTANT: TaskSpecificCircuit (Layer 0 provenance) is DISJOINT from
            # ComputationalPattern (Layer 2 learned). A circuit cannot BE both,
            # but it CAN have both (via rdf:type vs hasComputationalPattern)
            "DisjointClasses(TaskSpecificCircuit, ComputationalPattern)",

            # ========================================
            # FUNCTIONAL PROPERTIES (single-valued)
            # ========================================
            # Each circuit has at most one semantic specification
            "FunctionalObjectProperty(hasSemanticSpec)",

            # Each circuit has at most one architectural signature
            "FunctionalObjectProperty(hasArchitecturalSignature)",

            # Each circuit discovered in at most one experiment
            "FunctionalObjectProperty(discoveredIn)",

            # Each circuit from at most one model
            "FunctionalObjectProperty(fromModel)",

            # Each circuit at most one checkpoint
            "FunctionalObjectProperty(atCheckpoint)",

            # Each circuit belongs to at most one task family
            "FunctionalObjectProperty(belongsToTaskFamily)",

            # Metrics are single-valued
            "FunctionalDataProperty(confidence)",
            "FunctionalDataProperty(faithfulness)",
            "FunctionalDataProperty(causalEffect)",
            "FunctionalDataProperty(baselinePerformance)",
            "FunctionalDataProperty(circuitPerformance)",

            # ========================================
            # CARDINALITY CONSTRAINTS
            # ========================================
            # Every circuit must have at least one component
            "Circuit ⊑ ∃hasComponent.Component",

            # Every circuit must be discovered in some experiment
            "Circuit ⊑ ∃discoveredIn.Experiment",

            # Every circuit must belong to some task family
            "Circuit ⊑ ∃belongsToTaskFamily.TaskFamily",

            # Every edge must have exactly one source and one target
            "Edge ⊑ (=1 hasSource.Node) ⊓ (=1 hasTarget.Node)",

            # Every learned specification must have confidence score
            "LearnedSpecification ⊑ ∃confidence.xsd:float",

            # Every semantic spec must have predicates
            "SemanticSpec ⊑ ∃predicates.xsd:string",

            # ========================================
            # DOMAIN/RANGE CONSTRAINTS (type safety)
            # ========================================
            # Object property domains and ranges
            "hasComponent domain Circuit",
            "hasComponent range Component",

            "hasMemberNode domain Circuit",
            "hasMemberNode range Node",

            "hasMemberEdge domain Circuit",
            "hasMemberEdge range Edge",

            "connectsTo domain Component",
            "connectsTo range Component",

            "hasSource domain Edge",
            "hasSource range Node",

            "hasTarget domain Edge",
            "hasTarget range Node",

            "hasSemanticSpec domain Circuit",
            "hasSemanticSpec range SemanticSpec",

            "hasArchitecturalSignature domain Circuit",
            "hasArchitecturalSignature range ArchitecturalSignature",

            "hasComputationalPattern domain Circuit",
            "hasComputationalPattern range ComputationalPattern",

            "discoveredIn domain Circuit",
            "discoveredIn range Experiment",

            "fromModel domain Circuit",
            "fromModel range Model",

            "belongsToTaskFamily domain Circuit",
            "belongsToTaskFamily range TaskFamily",

            # Data property ranges
            "confidence range xsd:float",
            "faithfulness range xsd:float",
            "causalEffect range xsd:float",
            "baselinePerformance range xsd:float",
            "circuitPerformance range xsd:float",

            "inLayer range xsd:integer",
            "head range xsd:integer",
            "score range xsd:float",
            "rank range xsd:integer",

            "numClauses range xsd:integer",
            "avgClauseDepth range xsd:float",
            "avgClauseLength range xsd:float",
            "complexity range xsd:float",

            "hasRelativeDepth range xsd:float",
            "depthMin range xsd:float",
            "depthMax range xsd:float",
            "depthMean range xsd:float",

            "numAttentionHeads range xsd:integer",
            "numMLPBlocks range xsd:integer",

            # ========================================
            # ADDITIONAL SEMANTIC CONSTRAINTS
            # ========================================
            # Relative depth must be in [0,1]
            # Note: OWL 2 doesn't support numeric range restrictions directly
            # This would need to be validated in application code or SHACL
            # "hasRelativeDepth range xsd:float[>= 0.0, <= 1.0]",  # Not valid OWL 2

            # Confidence must be in [0,1]
            # "confidence range xsd:float[>= 0.0, <= 1.0]",  # Not valid OWL 2

            # NOTE: For numeric range validation (depth ∈ [0,1], confidence ∈ [0,1]),
            # use SHACL shapes or implement in extraction/validation code.
        ]

    # ========================================
    # HELPER METHODS
    # ========================================

    def get_circuit_type(self, task_family: str, role_or_phenomenon: str) -> URIRef:
        """
        Map (task, role) to circuit type.
        This is KNOWN from experimental design.

        Lookup is case-insensitive on role_or_phenomenon so that
        "Location", "LOCATION", and "location" all resolve correctly.
        """
        # Canonical map uses lowercase role keys for case-insensitive lookup
        type_map = {
            ('SemanticRoles', 'instrument'): self.namespace.InstrumentCircuit,
            ('SemanticRoles', 'location'): self.namespace.LocationCircuit,
            ('SemanticRoles', 'beneficiary'): self.namespace.BeneficiaryCircuit,
            ('SemanticRoles', 'agent'): self.namespace.AgentCircuit,
            ('SemanticRoles', 'patient'): self.namespace.PatientCircuit,
            ('SemanticRoles', 'theme'): self.namespace.ThemeCircuit,
            ('SemanticRoles', 'time'): self.namespace.TimeCircuit,
            ('SemanticRoles', 'manner'): self.namespace.MannerCircuit,
            ('SemanticRoles', 'purpose'): self.namespace.PurposeCircuit,
            ('SemanticRoles', 'path'): self.namespace.RoleCircuit,  # no PathCircuit yet

            ('IOI', 'ioi'): self.namespace.IOICircuit,
            ('IOI', 'namemover'): self.namespace.NameMoverCircuit,
            ('IOI', 'duplicatetoken'): self.namespace.DuplicateTokenCircuit,
            ('IOI', 'sinhibition'): self.namespace.SInhibitionCircuit,

            ('NumericalComparison', 'magnitudecomparison'): self.namespace.MagnitudeComparisonCircuit,
            ('NumericalComparison', 'digitextraction'): self.namespace.DigitExtractionCircuit,
            ('GreaterThan', 'magnitudecomparison'): self.namespace.MagnitudeComparisonCircuit,
        }

        key = (task_family, role_or_phenomenon.lower())
        return type_map.get(key, self.namespace.Circuit)

    def infer_computational_pattern(self, predicates: str) -> URIRef:
        """
        Infer computational pattern from learned predicates.
        This is LEARNED from ILP_sem output.
        """
        predicates_lower = predicates.lower()

        if 'binds' in predicates_lower or 'bind' in predicates_lower:
            return self.namespace.Binding
        elif 'compares' in predicates_lower or 'compare' in predicates_lower:
            return self.namespace.Comparison
        elif 'selects' in predicates_lower or 'select' in predicates_lower:
            return self.namespace.Selection
        elif 'inhibits' in predicates_lower or 'inhibit' in predicates_lower:
            return self.namespace.Inhibition
        elif 'copies' in predicates_lower or 'copy' in predicates_lower:
            return self.namespace.Copy
        elif 'extracts' in predicates_lower or 'extract' in predicates_lower:
            return self.namespace.Extraction
        else:
            return self.namespace.ComputationalPattern  # Generic

    def get_component_type_for_node(self, node_name: str) -> URIRef:
        """Map node name to appropriate component type"""
        if node_name == 'input':
            return self.namespace.Input
        elif node_name == 'logits':
            return self.namespace.Output  #
        elif node_name.startswith('a'):  # attention head pattern: a{layer}.h{head}
            return self.namespace.AttentionHead  #
        elif node_name.startswith('m'):  # MLP pattern: m{layer}
            return self.namespace.MLPBlock  # -
        else:
            return self.namespace.Component

    def parse_node_structure(self, node_name: str) -> Dict:
        """Extract layer/head from node name (e.g., 'a2.h3' → layer=2, head=3)"""
        result = {}

        if node_name.startswith('a'):  # attention: a2.h3
            parts = node_name.split('.')
            if len(parts) == 2:
                try:
                    result['layer'] = int(parts[0][1:])
                    result['head'] = int(parts[1][1:])
                except ValueError:
                    pass
        elif node_name.startswith('m'):  # MLP: m5
            try:
                result['layer'] = int(node_name[1:])
            except ValueError:
                pass

        return result

    # ========================================
    # VALIDATION HELPERS
    # ========================================

    def validate_property_usage(self, subject_type: str, property_name: str,
                                object_type: str = None) -> Dict[str, Any]:
        """
        Validate that a property is being used with correct domain/range.

        Args:
            subject_type: Type of the subject (e.g., 'Circuit')
            property_name: Property being used (e.g., 'hasComponent')
            object_type: Type of the object (optional, for additional checking)

        Returns:
            Dict with keys: 'valid' (bool), 'expected_domain', 'expected_range', 'message'

        Example:
            > schema.validate_property_usage('Circuit', 'hasComponent', 'Component')
            {'valid': True, 'expected_domain': 'Circuit', 'expected_range': 'Component'}

            > schema.validate_property_usage('Edge', 'hasComponent')
            {'valid': False, 'expected_domain': 'Circuit', 'expected_range': 'Component',
             'message': 'Domain mismatch: Edge used but Circuit expected'}
        """
        # Check object properties
        obj_sigs = self.get_property_signatures()
        if property_name in obj_sigs:
            sig = obj_sigs[property_name]
            expected_domain = sig['domain']
            expected_range = sig['range']

            if subject_type != expected_domain:
                return {
                    'valid': False,
                    'expected_domain': expected_domain,
                    'expected_range': expected_range,
                    'message': f"Domain mismatch: {subject_type} used but {expected_domain} expected"
                }

            if object_type and object_type != expected_range:
                return {
                    'valid': False,
                    'expected_domain': expected_domain,
                    'expected_range': expected_range,
                    'message': f"Range mismatch: {object_type} used but {expected_range} expected"
                }

            return {
                'valid': True,
                'expected_domain': expected_domain,
                'expected_range': expected_range
            }

        # Check data properties
        data_sigs = self.get_datatype_signatures()
        if property_name in data_sigs:
            sig = data_sigs[property_name]
            expected_domain = sig['domain']
            expected_range = sig['range']

            if subject_type != expected_domain:
                return {
                    'valid': False,
                    'expected_domain': expected_domain,
                    'expected_range': expected_range,
                    'message': f"Domain mismatch: {subject_type} used but {expected_domain} expected"
                }

            return {
                'valid': True,
                'expected_domain': expected_domain,
                'expected_range': expected_range
            }

        # Property not found
        return {
            'valid': False,
            'message': f"Property '{property_name}' not found in schema"
        }

    def get_admissible_universe(self, predicate_name: str, argument_position: int = 0) -> Set[str]:
        """
        Get admissible universe for a predicate argument (see App A.2 requirement).
        Used for bounded closed-world completion.

        Args:
            predicate_name: Name of the predicate (e.g., 'hasComponent')
            argument_position: 0 for domain (subject), 1 for range (object)

        Returns:
            Set of class names that are admissible for this argument position

        Example:
            > schema.get_admissible_universe('hasComponent', 0)
            {'Circuit'}  # Domain: only Circuits can have components

            > schema.get_admissible_universe('hasComponent', 1)
            {'Component', 'AttentionHead', 'MLPBlock', ...}  # Range: all Component subclasses
        """
        # Get signature
        obj_sigs = self.get_property_signatures()
        data_sigs = self.get_datatype_signatures()

        sig = obj_sigs.get(predicate_name) or data_sigs.get(predicate_name)
        if not sig:
            return set()

        if argument_position == 0:
            # Domain (subject)
            base_class = sig['domain']
        elif argument_position == 1:
            # Range (object)
            base_class = sig['range']
        else:
            return set()

        # If it's a datatype, return as-is
        if base_class.startswith('xsd:'):
            return {base_class}

        # Otherwise, get class hierarchy
        admissible = {base_class}

        # Add known subclasses based on inference axioms
        subclass_map = {
            'Component': {'AttentionHead', 'MLPBlock', 'LayerNorm', 'Residual', 'Input', 'Output'},
            'TaskSpecificCircuit': {'RoleCircuit', 'IOICircuit', 'NumericalCircuit',
                                    'InstrumentCircuit', 'LocationCircuit', 'BeneficiaryCircuit',
                                    'AgentCircuit', 'PatientCircuit', 'ThemeCircuit',
                                    'TimeCircuit', 'MannerCircuit', 'PurposeCircuit',
                                    'NameMoverCircuit', 'DuplicateTokenCircuit', 'SInhibitionCircuit',
                                    'MagnitudeComparisonCircuit', 'DigitExtractionCircuit'},
            'RoleCircuit': {'InstrumentCircuit', 'LocationCircuit', 'BeneficiaryCircuit',
                            'AgentCircuit', 'PatientCircuit', 'ThemeCircuit',
                            'TimeCircuit', 'MannerCircuit', 'PurposeCircuit'},
            'IOICircuit': {'NameMoverCircuit', 'DuplicateTokenCircuit', 'SInhibitionCircuit'},
            'NumericalCircuit': {'MagnitudeComparisonCircuit', 'DigitExtractionCircuit'},
            'ComputationalPattern': {'BindingPattern', 'ComparisonPattern', 'SelectionPattern',
                                     'InhibitionPattern', 'CopyPattern', 'ExtractionPattern'},
            'LearnedSpecification': {'SemanticSpec', 'ArchitecturalSpec'},
        }

        if base_class in subclass_map:
            admissible.update(subclass_map[base_class])

        return admissible


# Global instance
MECHINT_SCHEMA = MechIntSchema()

__all__ = ['MECHINT', 'MECHINT_BASE', 'MECHINT_SCHEMA', 'MechIntSchema']