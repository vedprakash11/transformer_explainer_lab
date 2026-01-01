"""
Explainability module for analyzing token relationships and coreference resolution.

This module identifies relationships between tokens using attention patterns,
such as pronoun-antecedent relationships, entity co-references, and semantic connections.
Also includes demo attention computation with human-readable explanations via Groq API.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Set, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import defaultdict
import os

# Try to import Groq, but make it optional
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None


def get_groq_client():
    """Get Groq client if API key is available."""
    if not GROQ_AVAILABLE:
        return None
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return None
    return Groq(api_key=groq_api_key)


def groq_generate_explanation(prompt: str) -> str:
    """
    Generate human-readable explanation using Groq API.
    
    Args:
        prompt: The prompt to send to Groq API
    
    Returns:
        Human-readable explanation string
    """
    client = get_groq_client()
    if not client:
        return "[Groq API not available] Please set GROQ_API_KEY environment variable to enable AI-generated explanations."
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in natural language processing and transformer models. Explain attention patterns, token relationships, and coreference resolution in a clear, concise, and human-readable way."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as ex:
        return f"[Groq API Error] {str(ex)}"


def demo_attention_on_sentence(
    sentence: str,
    embedding_dim: int = 8,
    use_groq: bool = True
) -> Dict[str, Any]:
    """
    Demonstrate self-attention on a dynamic sentence using dummy embeddings.
    This shows how attention works step-by-step with human-readable explanations.

    Args:
        sentence: Input sentence string
        embedding_dim: Dimension of the dummy embeddings
        use_groq: Whether to use Groq API for explanations

    Returns:
        Dictionary containing tokens, embeddings, Q/K/V matrices, attention weights,
        output vectors, pronoun-specific attention, and human-readable explanations
    """
    # -----------------------------
    # 1. Tokenize sentence
    # -----------------------------
    tokens = sentence.split()

    # -----------------------------
    # 2. Create dummy embeddings
    # -----------------------------
    np.random.seed(42)  # For reproducibility
    unique_tokens = list(set(tokens))
    embeddings_dict = {word: np.random.rand(embedding_dim) for word in unique_tokens}
    X = np.array([embeddings_dict[word] for word in tokens])

    # -----------------------------
    # 3. Create Q, K, V matrices
    # -----------------------------
    W_Q = np.random.rand(embedding_dim, embedding_dim)
    W_K = np.random.rand(embedding_dim, embedding_dim)
    W_V = np.random.rand(embedding_dim, embedding_dim)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # -----------------------------
    # 4. Scaled Dot-Product Attention
    # -----------------------------
    dk = K.shape[1]

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    scores = Q @ K.T / np.sqrt(dk)
    attention_weights = np.array([softmax(row) for row in scores])
    output = attention_weights @ V

    # -----------------------------
    # 5. Pronoun-specific attention analysis
    # -----------------------------
    pronouns = {"he", "him", "she", "her", "I", "me", "you", "they", "them", "it"}
    pronoun_attention = {}
    human_readable = {}
    pronoun_explanations = {}

    for pronoun in pronouns:
        if pronoun in tokens:
            idx = tokens.index(pronoun)
            pronoun_attention[pronoun] = attention_weights[idx]
            
            # Get top attended tokens for this pronoun
            top_indices = np.argsort(attention_weights[idx])[::-1][:5]
            top_tokens_with_scores = [
                (tokens[i], float(attention_weights[idx][i]))
                for i in top_indices
            ]
            
            # Generate human-readable explanation
            if use_groq:
                prompt = f"""Explain the attention pattern for the pronoun '{pronoun}' in the sentence: "{sentence}"

The pronoun '{pronoun}' has the following attention scores to other tokens:
{', '.join([f'{token} ({score:.3f})' for token, score in top_tokens_with_scores])}

Explain:
1. Which tokens the pronoun '{pronoun}' is paying most attention to
2. Why these attention patterns make sense linguistically
3. What relationships or coreference these patterns suggest
4. Keep the explanation concise (2-3 sentences)."""
                
                explanation = groq_generate_explanation(prompt)
                pronoun_explanations[pronoun] = explanation
            else:
                pronoun_explanations[pronoun] = f"Pronoun '{pronoun}' attends most to: {', '.join([t for t, _ in top_tokens_with_scores[:3]])}"

    # Generate overall sentence explanation
    if use_groq:
        overall_prompt = f"""Analyze the self-attention patterns in this sentence: "{sentence}"

Tokens: {', '.join(tokens)}

Explain:
1. The overall attention structure of this sentence
2. Key relationships between tokens based on attention weights
3. Any pronoun-antecedent relationships you can identify
4. What the attention patterns reveal about how the model processes this sentence

Keep the explanation clear and concise (3-4 sentences)."""
        overall_explanation = groq_generate_explanation(overall_prompt)
    else:
        overall_explanation = f"Attention analysis for sentence: {sentence}"

    # Return all results in a structured dict
    return {
        "tokens": tokens,
        "embeddings": X,
        "Q": Q,
        "K": K,
        "V": V,
        "attention_weights": attention_weights,
        "output_vectors": output,
        "pronoun_attention": pronoun_attention,
        "pronoun_explanations": pronoun_explanations,
        "overall_explanation": overall_explanation,
        "sentence": sentence,
    }


def identify_coreference_chains(
    attentions: Tuple[torch.Tensor, ...],
    tokens: List[str],
    threshold: float = 0.3,
    remove_special_tokens: bool = True
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identify coreference chains using attention patterns.
    
    Args:
        attentions: Tuple of attention tensors from all layers
        tokens: List of token strings
        threshold: Minimum attention threshold for considering a relationship
        remove_special_tokens: Whether to exclude [CLS] and [SEP] tokens
    
    Returns:
        Dictionary mapping each token to list of related tokens with attention scores
    """
    # Filter out special tokens if requested
    if remove_special_tokens:
        valid_indices = [i for i, token in enumerate(tokens) 
                        if token not in ['[CLS]', '[SEP]', '<s>', '</s>']]
        filtered_tokens = [tokens[i] for i in valid_indices]
    else:
        valid_indices = list(range(len(tokens)))
        filtered_tokens = tokens
    
    # Aggregate attention across all layers and heads
    all_attentions = torch.stack(attentions)  # (layers, batch, heads, seq, seq)
    avg_attention = all_attentions.mean(dim=(0, 2))[0]  # Average over layers and heads
    
    # Filter attention matrix to only include valid tokens
    filtered_attention = avg_attention[valid_indices][:, valid_indices]
    
    relationships = defaultdict(list)
    seen_pairs = set()  # To avoid duplicate entries
    
    for i, token_i in enumerate(filtered_tokens):
        for j, token_j in enumerate(filtered_tokens):
            if i != j:
                # Bidirectional attention (i->j and j->i)
                attn_ij = filtered_attention[i, j].item()
                attn_ji = filtered_attention[j, i].item()
                avg_attn = (attn_ij + attn_ji) / 2
                
                if avg_attn > threshold:
                    # Add bidirectional relationships to ensure both directions are captured
                    pair_key = tuple(sorted([token_i, token_j]))
                    if pair_key not in seen_pairs:
                        relationships[token_i].append((token_j, avg_attn))
                        relationships[token_j].append((token_i, avg_attn))
                        seen_pairs.add(pair_key)
    
    # Sort by attention score
    for token in relationships:
        relationships[token].sort(key=lambda x: x[1], reverse=True)
    
    return dict(relationships)


def identify_pronoun_antecedents(
    tokens: List[str],
    relationships: Dict[str, List[Tuple[str, float]]]
) -> List[Tuple[str, str, float]]:
    """
    Identify pronoun-antecedent relationships.
    
    Args:
        tokens: List of token strings
        relationships: Dictionary of token relationships
    
    Returns:
        List of (pronoun, antecedent, confidence) tuples
    """
    pronouns = {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'their', 'theirs', 'I', 'me', 'you'}
    pronoun_antecedents = []
    seen_pairs = set()  # To avoid duplicates
    
    for token in tokens:
        token_lower = token.lower().strip('.,!?;:')
        if token_lower in pronouns:
            # Find potential antecedents (usually nouns or proper nouns)
            for related_token, score in relationships.get(token, []):
                related_lower = related_token.lower().strip('.,!?;:')
                # Skip if related token is also a pronoun
                if related_lower not in pronouns:
                    pair = (token, related_token)
                    if pair not in seen_pairs:
                        pronoun_antecedents.append((token, related_token, score))
                        seen_pairs.add(pair)
            
            # Also check reverse: if antecedent has pronoun in its relationships
            for other_token in tokens:
                if other_token != token:
                    other_lower = other_token.lower().strip('.,!?;:')
                    if other_lower not in pronouns:
                        # Check if other_token has this pronoun in its relationships
                        for related_token, score in relationships.get(other_token, []):
                            if related_token == token:
                                pair = (token, other_token)
                                if pair not in seen_pairs:
                                    pronoun_antecedents.append((token, other_token, score))
                                    seen_pairs.add(pair)
    
    return pronoun_antecedents


def identify_entity_relationships(
    tokens: List[str],
    relationships: Dict[str, List[Tuple[str, float]]],
    min_attention: float = 0.25
) -> List[Tuple[str, str, str, float]]:
    """
    Identify relationships between entities (people, objects, etc.).
    
    Args:
        tokens: List of token strings
        relationships: Dictionary of token relationships
        min_attention: Minimum attention score for relationship
    
    Returns:
        List of (entity1, entity2, relationship_type, confidence) tuples
    """
    entity_relations = []
    
    # Common relationship indicators
    action_verbs = {'ordered', 'brought', 'gave', 'took', 'bought', 'made', 'did', 'said', 'went'}
    possessive_indicators = {'for', 'to', 'with', 'by', 'from'}
    
    for token_i, related_list in relationships.items():
        for token_j, score in related_list:
            if score < min_attention:
                continue
            
            # Check for action relationships
            token_i_lower = token_i.lower().strip('.,!?;:')
            token_j_lower = token_j.lower().strip('.,!?;:')
            
            # Find relationship type based on context
            rel_type = "related"
            
            # Check if there's a verb between them indicating action
            token_i_idx = tokens.index(token_i) if token_i in tokens else -1
            token_j_idx = tokens.index(token_j) if token_j in tokens else -1
            
            if token_i_idx >= 0 and token_j_idx >= 0:
                # Check tokens between them
                start_idx = min(token_i_idx, token_j_idx)
                end_idx = max(token_i_idx, token_j_idx)
                between_tokens = tokens[start_idx:end_idx+1]
                
                for bt in between_tokens:
                    bt_lower = bt.lower().strip('.,!?;:')
                    if bt_lower in action_verbs:
                        rel_type = "action"
                    elif bt_lower in possessive_indicators:
                        rel_type = "possession"
            
            entity_relations.append((token_i, token_j, rel_type, score))
    
    return entity_relations


def visualize_relationships(
    tokens: List[str],
    relationships: Dict[str, List[Tuple[str, float]]],
    pronoun_antecedents: List[Tuple[str, str, float]],
    entity_relations: List[Tuple[str, str, str, float]],
    remove_special_tokens: bool = True
) -> go.Figure:
    """
    Create a network graph visualization of token relationships.
    
    Args:
        tokens: List of token strings
        relationships: Dictionary of token relationships
        pronoun_antecedents: List of pronoun-antecedent pairs
        entity_relations: List of entity relationships
        remove_special_tokens: Whether to exclude [CLS] and [SEP] tokens
    
    Returns:
        Plotly Figure with network graph
    """
    # Filter out special tokens if requested
    if remove_special_tokens:
        filtered_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '<s>', '</s>']]
    else:
        filtered_tokens = tokens
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes (only from filtered tokens)
    for token in filtered_tokens:
        G.add_node(token)
    
    # Add edges with weights
    edge_traces = []
    
    # Add pronoun-antecedent edges (highlighted)
    # Also create transitive relationships: if multiple pronouns refer to same antecedent,
    # link them together
    antecedent_to_pronouns = defaultdict(list)
    for pronoun, antecedent, score in pronoun_antecedents:
        if pronoun in G and antecedent in G:
            G.add_edge(pronoun, antecedent, weight=score, type='coreference')
            antecedent_to_pronouns[antecedent].append((pronoun, score))
    
    # Link pronouns that refer to the same antecedent (transitive relationships)
    for antecedent, pronouns_list in antecedent_to_pronouns.items():
        if len(pronouns_list) > 1:
            # Link all pronouns that refer to the same antecedent
            for i, (pronoun1, score1) in enumerate(pronouns_list):
                for pronoun2, score2 in pronouns_list[i+1:]:
                    if pronoun1 in G and pronoun2 in G:
                        # Use average score for the transitive link
                        avg_score = (score1 + score2) / 2
                        G.add_edge(pronoun1, pronoun2, weight=avg_score, type='coreference_chain')
    
    # Also add relationships from the relationships dictionary that might have been missed
    # This ensures bidirectional relationships are captured (e.g., Nitin ↔ him)
    for token, related_list in relationships.items():
        if token in G:
            for related_token, score in related_list:
                if related_token in G and score > 0.2:
                    # Only add if edge doesn't exist or if this score is higher
                    if not G.has_edge(token, related_token):
                        G.add_edge(token, related_token, weight=score, type='attention')
                    elif G[token][related_token].get('weight', 0) < score:
                        G[token][related_token]['weight'] = score
    
    # Add entity relationship edges
    for entity1, entity2, rel_type, score in entity_relations:
        if entity1 in G and entity2 in G and score > 0.2:
            if not G.has_edge(entity1, entity2):
                G.add_edge(entity1, entity2, weight=score, type=rel_type)
            elif G[entity1][entity2].get('weight', 0) < score:
                G[entity1][entity2]['weight'] = score
                G[entity1][entity2]['type'] = rel_type
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Extract node positions
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_info.append({
            'from': edge[0],
            'to': edge[1],
            'weight': edge[2].get('weight', 0),
            'type': edge[2].get('type', 'related')
        })
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        # Color based on type
        if any(node == p[0] for p in pronoun_antecedents):
            node_colors.append('lightcoral')  # Pronouns
        elif any(node == p[1] for p in pronoun_antecedents):
            node_colors.append('lightblue')  # Antecedents
        else:
            node_colors.append('lightgreen')  # Other tokens
        
        # Size based on degree
        degree = G.degree(node)
        node_sizes.append(20 + degree * 5)
        node_text.append(node)
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='black')
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10, color='black'),
        hoverinfo='text',
        hovertext=[f"{node}<br>Connections: {G.degree(node)}" for node in G.nodes()],
        showlegend=False
    ))
    
    fig.update_layout(
        title="Token Relationship Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )
    
    return fig


def create_relationship_table(
    pronoun_antecedents: List[Tuple[str, str, float]],
    entity_relations: List[Tuple[str, str, str, float]]
) -> List[Dict[str, any]]:
    """
    Create a table of identified relationships.
    
    Args:
        pronoun_antecedents: List of pronoun-antecedent pairs
        entity_relations: List of entity relationships
    
    Returns:
        List of dictionaries for table display
    """
    table_data = []
    
    # Add pronoun-antecedent relationships
    for pronoun, antecedent, score in pronoun_antecedents:
        table_data.append({
            "Token 1": pronoun,
            "Token 2": antecedent,
            "Relationship": "Coreference (Pronoun-Antecedent)",
            "Confidence": f"{score:.3f}",
            "Explanation": f"'{pronoun}' refers to '{antecedent}'"
        })
    
    # Add entity relationships (top ones)
    entity_relations_sorted = sorted(entity_relations, key=lambda x: x[3], reverse=True)[:20]
    for entity1, entity2, rel_type, score in entity_relations_sorted:
        table_data.append({
            "Token 1": entity1,
            "Token 2": entity2,
            "Relationship": rel_type.title(),
            "Confidence": f"{score:.3f}",
            "Explanation": f"'{entity1}' and '{entity2}' are related through {rel_type}"
        })
    
    return table_data
import spacy
import networkx as nx
import plotly.graph_objects as go
from typing import List, Tuple
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")


def extract_entities_and_relation(sentence: str):
    """Extract (subject, relation, object) from sentence."""
    doc = nlp(sentence)

    subject = ""
    obj = ""
    relation = ""

    for token in doc:
        if "subj" in token.dep_:
            subject = token.text
        if "obj" in token.dep_:
            obj = token.text
        if token.dep_ == "ROOT":
            relation = token.lemma_

    return subject, relation, obj
def build_knowledge_graph(sentence: str):
    subj, rel, obj = extract_entities_and_relation(sentence)

    G = nx.DiGraph()

    if subj and obj:
        G.add_node(subj, type="entity")
        G.add_node(obj, type="entity")
        G.add_edge(subj, obj, label=rel)

    return G


import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")


def build_dynamic_knowledge_graph(sentence: str):
    """
    Builds a semantic knowledge graph from any sentence.
    Handles:
    - Multiple subjects
    - Multiple actions
    - Objects
    - Temporal info
    - Beneficiaries (for/to)
    """

    doc = nlp(sentence)
    G = nx.DiGraph()

    current_subjects = []
    current_verb = None
    last_verb = None

    for token in doc:

        # --- SUBJECTS ---
        if token.dep_ in ("nsubj", "nsubjpass"):
            subj = token.text
            current_subjects.append(subj)
            G.add_node(subj, type="entity")

        # --- VERB / ACTION ---
        if token.dep_ == "ROOT" or token.pos_ == "VERB":
            current_verb = token.lemma_
            last_verb = current_verb
            G.add_node(current_verb, type="action")

        # --- OBJECT ---
        if token.dep_ in ("dobj", "obj"):
            obj = token.text
            G.add_node(obj, type="entity")

            for subj in current_subjects:
                G.add_edge(subj, obj, relation=current_verb)

        # --- TEMPORAL INFO ---
        if token.ent_type_ in ("DATE", "TIME"):
            time_node = token.text
            G.add_node(time_node, type="time")
            if last_verb:
                G.add_edge(last_verb, time_node, relation="time")

        # --- PREPOSITION OBJECTS (for / to) ---
        if token.dep_ == "pobj":
            pobj = token.text
            prep = token.head.text

            G.add_node(pobj, type="entity")

            if last_verb:
                if prep in ["for", "to"]:
                    G.add_edge(last_verb, pobj, relation="beneficiary")
                else:
                    G.add_edge(last_verb, pobj, relation=prep)

    return G
import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")


def build_semantic_graph(sentence: str):
    doc = nlp(sentence)
    G = nx.DiGraph()

    subjects = []
    current_verb = None
    current_object = None
    time_info = None

    for token in doc:

        # SUBJECT
        if token.dep_ in ("nsubj", "nsubjpass"):
            subjects.append(token.text)
            G.add_node(token.text, type="entity")

        # VERB
        if token.pos_ == "VERB":
            current_verb = token.lemma_

        # OBJECT
        if token.dep_ in ("dobj", "obj"):
            current_object = token.text
            G.add_node(current_object, type="entity")

            for subj in subjects:
                edge_label = f"{current_verb}({current_object})"
                G.add_edge(subj, current_object, label=edge_label)

        # TIME
        if token.ent_type_ == "DATE":
            time_info = token.text
            for subj in subjects:
                if current_verb:
                    G.add_edge(
                        subj,
                        token.text,
                        label=f"{current_verb}(time)"
                    )

        # PREPOSITIONAL OBJECTS (for / to)
        if token.dep_ == "pobj":
            prep = token.head.text
            obj = token.text
            G.add_node(obj, type="entity")

            if current_verb:
                G.add_edge(
                    current_verb,
                    obj,
                    label=f"{prep}"
                )

    return G
import plotly.graph_objects as go

def visualize_graph(sentence: str):
    G = build_semantic_graph(sentence)
    pos = nx.spring_layout(G, k=1.3)

    edge_x, edge_y, edge_text = [], [], []

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_text.append((mid_x, mid_y, d.get("label", "")))

    fig = go.Figure()

    # Draw edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="gray"),
        hoverinfo="none"
    ))

    # Draw edge labels
    for x, y, label in edge_text:
        fig.add_annotation(
            x=x,
            y=y,
            text=label,
            showarrow=False,
            font=dict(size=10, color="darkred")
        )

    # Draw nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=30, color="lightblue"),
        text=list(G.nodes()),
        textposition="middle center"
    ))

    fig.update_layout(
        title="Semantic Knowledge Graph (Edge-Based Relations)",
        showlegend=False,
        height=650
    )

    return fig

def visualize_knowledge_graph(sentence: str) -> go.Figure:
    G = build_dynamic_knowledge_graph(sentence)

    pos = nx.spring_layout(G, k=1.5)

    # Nodes
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    # Edges
    edge_x, edge_y, edge_text = [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(data.get("label", ""))

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="#888"),
        hoverinfo="none"
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=30, color="skyblue", line=dict(width=2)),
        text=node_text,
        textposition="middle center",
        hoverinfo="text"
    ))

    fig.update_layout(
        title="Knowledge Graph (Entity → Relation → Entity)",
        showlegend=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig


def explain_sentence(
    attentions: Tuple[torch.Tensor, ...],
    tokens: List[str],
    threshold: float = 0.3,
    remove_special_tokens: bool = True,
    use_groq: bool = True,
    sentence: str = None
) -> Dict[str, any]:
    """
    Main function to explain relationships in a sentence.
    
    Args:
        attentions: Tuple of attention tensors from all layers
        tokens: List of token strings
        threshold: Minimum attention threshold
        remove_special_tokens: Whether to exclude [CLS] and [SEP] tokens
        use_groq: Whether to use Groq API for human-readable explanations
    
    Returns:
        Dictionary containing all explainability results
    """
    # Filter tokens for display
    if remove_special_tokens:
        filtered_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '<s>', '</s>']]
    else:
        filtered_tokens = tokens
    
    # Identify relationships (this function now handles filtering internally)
    relationships = identify_coreference_chains(attentions, tokens, threshold, remove_special_tokens)
    
    # Identify pronoun-antecedent pairs (using filtered tokens)
    pronoun_antecedents = identify_pronoun_antecedents(filtered_tokens, relationships)
    
    # Identify entity relationships (using filtered tokens)
    entity_relations = identify_entity_relationships(filtered_tokens, relationships)
    
    # Create visualization (using filtered tokens)
    
    
    # Create table data
    table_data = create_relationship_table(pronoun_antecedents, entity_relations)
    
    # Generate human-readable explanation using Groq
    sentence_text = ' '.join(filtered_tokens)
    network_fig = visualize_graph(
        sentence
    )
    if use_groq:
        if pronoun_antecedents:
            groq_prompt = f"""Analyze the attention patterns and coreference resolution in this sentence: "{sentence_text}"

Identified pronoun-antecedent relationships:
{chr(10).join([f"- '{pronoun}' refers to '{antecedent}' (confidence: {score:.3f})" for pronoun, antecedent, score in pronoun_antecedents[:5]])}

Explain in 2-3 clear sentences:
1. What these pronoun-antecedent relationships mean
2. Why the attention patterns support these relationships
3. What this tells us about how the model understands the sentence"""
            
            groq_explanation = groq_generate_explanation(groq_prompt)
        else:
            # Even without pronoun-antecedents, we can still analyze the sentence
            groq_prompt = f"""Analyze the attention patterns and token relationships in this sentence: "{sentence_text}"

Explain in 2-3 clear sentences:
1. What the attention patterns reveal about how tokens relate to each other
2. Key relationships or connections between tokens
3. What this tells us about how the model processes this sentence"""
            
            groq_explanation = groq_generate_explanation(groq_prompt)
    else:
        groq_explanation = None  # Don't show a message if Groq is explicitly disabled
    
    return {
        'relationships': relationships,
        'pronoun_antecedents': pronoun_antecedents,
        'entity_relations': entity_relations,
        'network_figure': network_fig,
        'table_data': table_data,
        'groq_explanation': groq_explanation,
    }
