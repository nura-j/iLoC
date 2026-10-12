"""
CFS storage — persist CFS results to TTL (Layer 2), JSON, and Prolog facts.

Supports three output formats:
    1. TTL triples (via OWL exporter) — Layer 2 of the CKG
    2. JSON (via results_export) — layer2.cfs section
    3. Prolog facts (.pl file) — background knowledge for ILP_arch
"""


import json

from rdflib import Graph as RDFGraph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from .cfs import CFSResult
from .dla import LogitLensResult


def add_cfs_to_ttl(
    circuit_ttl_path: str,
    cfs: CFSResult,
    namespace=None,
) -> str:
    """
    Add CFS triples to an existing circuit TTL file (Layer 2).

    Adds per-component:
        - mechint:dlaScore (float)
        - mechint:causallyMarginal (bool)
        - mechint:attendsTo with role and weight

    Args:
        circuit_ttl_path: path to existing .ttl file
        cfs: computed CFSResult
        namespace: rdflib Namespace (default: mechint)

    Returns:
        path to updated TTL file
    """
    from ..ontology.schema import MECHINT

    if namespace is None:
        namespace = MECHINT

    rdf = RDFGraph()
    rdf.parse(circuit_ttl_path, format="turtle")
    rdf.bind("mechint", namespace)

    # Find the circuit URI
    circuit_uri = None
    for s, _, _ in rdf.triples((None, RDF.type, namespace.Circuit)):
        circuit_uri = s
        break
    if circuit_uri is None:
        raise ValueError(f"No mechint:Circuit found in {circuit_ttl_path}")

    # Add CFS metadata to circuit
    rdf.add((circuit_uri, namespace.hasCFS, Literal(True, datatype=XSD.boolean)))
    rdf.add((circuit_uri, namespace.cfsNumExamples,
             Literal(cfs.num_examples, datatype=XSD.integer)))
    rdf.add((circuit_uri, namespace.cfsEpsilonDLA,
             Literal(cfs.epsilon_dla, datatype=XSD.float)))

    # Add per-component CFS data
    for name, comp in cfs.components.items():
        node_uri = _node_uri(namespace, name)

        # DLA score
        rdf.add((node_uri, namespace.dlaScore,
                 Literal(comp.dla_score, datatype=XSD.float)))

        # Causally marginal flag
        rdf.add((node_uri, namespace.causallyMarginal,
                 Literal(comp.causally_marginal, datatype=XSD.boolean)))

        # Attention profile: one triple per role
        for role, weight in comp.attention_profile.items():
            profile_uri = URIRef(f"{node_uri}/attn_profile_{role}")
            rdf.add((profile_uri, RDF.type, namespace.AttentionRoleProfile))
            rdf.add((profile_uri, namespace.tokenRole, Literal(role)))
            rdf.add((profile_uri, namespace.roleWeight,
                     Literal(weight, datatype=XSD.float)))
            rdf.add((node_uri, namespace.attendsTo, profile_uri))

    rdf.serialize(destination=circuit_ttl_path, format="turtle")
    return circuit_ttl_path


def save_cfs_json(
    cfs: CFSResult,
    out_path: str,
) -> str:
    """
    Save CFS as a standalone JSON file.

    Args:
        cfs: computed CFSResult
        out_path: output file path

    Returns:
        path to JSON file
    """
    with open(out_path, "w") as f:
        json.dump(cfs.to_dict(), f, indent=2)
    return out_path


def add_cfs_to_results_json(
    json_path: str,
    cfs: CFSResult,
) -> str:
    """
    Add CFS data to an existing results JSON file (layer2.cfs section).

    Args:
        json_path: path to existing *_results.json
        cfs: computed CFSResult

    Returns:
        path to updated JSON file
    """
    with open(json_path) as f:
        record = json.load(f)

    if record.get("layer2") is None:
        record["layer2"] = {}

    record["layer2"]["cfs"] = cfs.to_dict()

    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    return json_path


def save_cfs_prolog(
    cfs: CFSResult,
    out_path: str,
) -> str:
    """
    Save CFS as Prolog facts for ILP_arch background knowledge:
        dla_score(v, δ).
        attends_to(v, r, p).

    Args:
        cfs: computed CFSResult
        out_path: output .pl file path

    Returns:
        path to .pl file
    """
    facts = cfs.to_prolog_facts()
    header = [
        "% Causal Functional Signature (CFS) — Layer 2 facts",
        f"% epsilon_dla = {cfs.epsilon_dla}",
        f"% num_examples = {cfs.num_examples}",
        f"% components = {len(cfs.components)}",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(header + facts) + "\n")
    return out_path


def save_logit_lens_json(
    ll: LogitLensResult,
    out_path: str,
) -> str:
    """Save logit lens result as standalone JSON."""
    data = {
        "n_layers": ll.n_layers,
        "num_examples": ll.num_examples,
        "per_layer": ll.per_layer,
        "per_layer_dla": ll.per_layer_dla,
        "crossing_layer": ll.crossing_layer(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def _node_uri(namespace, node_name: str) -> URIRef:
    """Convert node name (e.g. 'a0.h3') to a URI."""
    safe = node_name.replace(".", "_")
    return URIRef(f"{namespace}node_{safe}")
