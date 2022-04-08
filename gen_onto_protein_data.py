import os
import numpy as np
import lmdb
import pickle as pkl
from Bio import SwissProt
from goatools.obo_parser import GODag, GOTerm


NODE_TYPE_MAPPING = {
    'biological_process': 'Process',
    'molecular_function': 'Function',
    'cellular_component': 'Component'
}


def create_goa_triplet(fin_path, fout_path, protein_path):
    print('Loading gene ontology annotation...')

    cnt = 0
    protein_set = set()
    goa_set = set()
    valid_protein_set = set()
    part_in_go_term_set = set()

    # load swissprot protein
    with open(protein_path, 'r') as handle:
        for rec in handle.readlines():
            protein_set.add(rec.rstrip('\n').split()[0])

    print('A0A023PZB3' in protein_set)
    
    if not os.path.exists(fout_path):
        os.mkdir(fout_path)
    out_component_handle = open(os.path.join(fout_path, 'component.txt'), 'w')
    out_function_handle = open(os.path.join(fout_path, 'function.txt'), 'w')
    out_process_handle = open(os.path.join(fout_path, 'process.txt'), 'w')

    for idx, line in enumerate(open(fin_path, 'r')):
        # skip annotation info.
        if idx < 9:
            continue

        # key field:
        # index 0: DB
        # index 1: DB object id (head entity)
        # index 3: Qualifier (relation)
        # index 4: GO id (tail entity)
        # index 6: evidence code
        # index 8: aspect (node type, e.g. {C, F, P})
        # index 11: DB object type (e.g. protein)
        rec = line.rstrip("\n").split("\t")
        
        if rec[0] != 'UniProtKB' or rec[11] != 'protein':
            continue
        
        if rec[1] in protein_set:
            goa = f'{rec[1]}_{rec[3]}_{rec[4]}'
            if goa not in goa_set:
                goa_set.add(goa)
                valid_protein_set.add(rec[1])
                part_in_go_term_set.add(rec[4])

                if rec[8] == 'C':
                    out_component_handle.write(f'{rec[1]} {rec[3]} {rec[4]} {rec[6]}\n')
                elif rec[8] == 'F':
                    out_function_handle.write(f'{rec[1]} {rec[3]} {rec[4]} {rec[6]}\n')
                elif rec[8] == 'P':
                    out_process_handle.write(f'{rec[1]} {rec[3]} {rec[4]} {rec[6]}\n')
                else:
                    raise Exception('the ontology type not supported.')
        
        if idx % 100000 == 0:
            print(f'the number of valid protein: {len(valid_protein_set)}')
            print(f'the number of involved go term: {len(part_in_go_term_set)}')
            print('-----------------------------------------------------')

    out_component_handle.close()
    out_function_handle.close()
    out_process_handle.close()

    print('Finished!')
    print(f'the number of valid protein: {len(valid_protein_set)}')
    print(f'the number of involved go term: {len(part_in_go_term_set)}')

def create_uniprot_data(fin_path, fout_path):
    total_protein = 0
    valid_protein_list = []

    with open(fout_path, 'w') as out_handle:
        with open(fin_path, 'r') as in_handle:
            for rec in SwissProt.parse(in_handle):
                if rec.sequence is not None:
                    out_handle.write(f"{rec.accessions[0]} {rec.sequence}\n")

    print('Finished!')

def create_go_data(fin_path, fout_graph_path, fout_detail_path, fout_leaf_path):
    print('Loading gene ontology term...')

    go_graph_handle = open(fout_graph_path, 'w')
    go_detail_handle = open(fout_detail_path, 'w')
    go_leaf_handle = open(fout_leaf_path, 'w')

    godag = GODag(fin_path, optional_attrs={'relationship'})
    go_onto_set = set()
    leaf_go_set = set()
    max_level = -1
    for go_id, go_term in godag.items():
        # deal current node's parents ('is_a')
        cur_node = go_id
        cur_node_type = NODE_TYPE_MAPPING[go_term.namespace]
        cur_node_name = go_term.name
        cur_node_desc = f'{cur_node_name}: {go_term.definition}'
        cur_node_level = go_term.level

        go_detail_handle.write(f'{cur_node}\t{cur_node_type}\t{cur_node_desc}\t{cur_node_level}\n')

        if cur_node_level > max_level:
            max_level = cur_node_level

        for parent in go_term.parents:
            oth_node= parent.id
            oth_node_type = NODE_TYPE_MAPPING[parent.namespace]

            # remove those node existing children nodes.
            if oth_node in leaf_go_set:
                leaf_go_set.remove(oth_node)

            triplet = f'{cur_node}-is_a-{oth_node}'
            if triplet not in go_onto_set:
                go_graph_handle.write(f'{cur_node} is_a {oth_node}\n')
                go_onto_set.add(triplet)

        # deal current node' children nodes (is_a).
        for child in go_term.children:
            oth_node = child.id
            oth_node_type = NODE_TYPE_MAPPING[child.namespace]

            triplet = f'{oth_node}-is_a-{cur_node}'
            if triplet not in go_onto_set:
                go_graph_handle.write(f'{oth_node} is_a {cur_node}\n')
                go_onto_set.add(triplet)
            
        # deal remain relationship
        if go_term.relationship:
            for r, terms in go_term.relationship.items():
                for term in terms:
                    oth_node = term.id
                    oth_node_type = NODE_TYPE_MAPPING[term.namespace]

                    triplet = f'{cur_node}-{r}-{oth_node}'
                    if triplet not in go_onto_set:
                        go_graph_handle.write(f'{cur_node} {r} {oth_node}\n')
                        go_onto_set.add(triplet)

        # temporarily saving current node which don't exist children nodes.
        if len(go_term.children) == 0:
            leaf_go_set.add(cur_node)

    for go_term in leaf_go_set:
        go_leaf_handle.write(f'{go_term}\n')

    go_graph_handle.close()
    go_detail_handle.close()
    go_leaf_handle.close()


def create_onto_protein_data(
    fin_go_graph_path,
    fin_go_detail_path,
    fin_goa_path,
    fin_protein_seq_path,
    fout_path
):
    if not os.path.exists(fout_path):
        os.mkdir(fout_path)

    # TODO: dataset split: transductive and inductive

    go2id = {}
    protein2id = {}
    relation2id = {}
    cur_relation_idx = 0
    go2id_handle = open(os.path.join(fout_path, 'go2id.txt'), 'w')
    protein2id_handle = open(os.path.join(fout_path, 'protein2id.txt'), 'w')
    relation2id_handle = open(os.path.join(fout_path, 'relation2id.txt'), 'w')
    go_def_handle = open(os.path.join(fout_path, 'go_def.txt'), 'w')
    go_type_handle = open(os.path.join(fout_path, 'go_type.txt'), 'w')
    protein_seq_handle = open(os.path.join(fout_path, 'protein_seq.txt'), 'w')
    go_go_triplet_handle = open(os.path.join(fout_path, 'go_go_triplet.txt'), 'w')
    protein_go_triplet_handle = open(os.path.join(fout_path, 'protein_go_triplet.txt'), 'w')

    with open(fin_go_detail_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            rec = line.rstrip('\n').split('\t')
            go_term_id = rec[0]
            go_term_def = rec[2]
            go_term_type = rec[1]

            go2id[go_term_id] = idx
            go_def_handle.write(f'{go_term_def}\n')
            go_type_handle.write(f'{go_term_type}\n')

    for go, id in go2id.items():
        go2id_handle.write(f'{go} {id}\n')

    go_def_handle.close()
    go_type_handle.close()
    go2id_handle.close()

    with open(fin_go_graph_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            rec = line.rstrip('\n').split()
            head, relation, tail = rec

            if relation not in relation2id:
                relation2id[relation] = cur_relation_idx
                cur_relation_idx += 1
            
            head_id = go2id[head]
            relation_id = relation2id[relation]
            tail_id = go2id[tail]

            go_go_triplet_handle.write(f'{head_id} {relation_id} {tail_id}\n')

    go_go_triplet_handle.close()
    
    with open(fin_protein_seq_path, 'r') as f:
        db_env = lmdb.open(os.path.join(fout_path, 'swiss_seq'), map_size=1099511627776)
        update_freq = 1e-5
        txn = db_env.begin(write=True)
        for idx, line in enumerate(f.readlines()):
            rec = line.rstrip('\n').split()
            protein, seq = rec

            protein2id[protein] = idx
            protein_seq_handle.write(f'{seq}\n')
            # save protein sequence to lmdb
            txn.put(str(idx).encode(), pkl.dumps(seq))
            if idx % update_freq == 0:
                txn.commit()
                txn = db_env.begin(write=True)

            txn.put('num_examples'.encode(), pkl.dumps(idx+1))
        txn.commit()
        db_env.close()
    
    for protein, id in protein2id.items():
        protein2id_handle.write(f'{protein} {id}\n')
    
    protein_seq_handle.close()
    protein2id_handle.close()

    for type in ['component.txt', 'function.txt', 'process.txt']:
        with open(os.path.join(fin_goa_path, type)) as f:
            for line in f.readlines():
                rec = line.rstrip('\n').split()
                protein, relation, go, _ = rec

                if relation not in relation2id:
                    relation2id[relation] = cur_relation_idx
                    cur_relation_idx += 1

                protein_id = protein2id[protein]
                relation_id = relation2id[relation]

                # filter triplet which go term don't exist in go.obo
                if go in go2id:
                    go_id = go2id[go]
                    protein_go_triplet_handle.write(f'{protein_id} {relation_id} {go_id}\n')
    
    for relation, id in relation2id.items():
        relation2id_handle.write(f'{relation} {id}\n')

    protein_go_triplet_handle.close()
    relation2id_handle.close()

if __name__ == '__main__':
    create_uniprot_data('data/original_data/uniprot_sprot.dat', 'data/onto_protein_data/protein_seq_map.txt')
    create_goa_triplet('data/original_data/goa_uniprot_all.gaf', 'data/onto_protein_data/protein_go_triplet', 'data/onto_protein_data/protein_seq_map.txt')

    create_go_data(
       fin_path='data/original_data/go.obo', 
       fout_graph_path='data/onto_protein_data/go_graph.txt', 
       fout_detail_path='data/onto_protein_data/go_detail.txt',
       fout_leaf_path='data/onto_protein_data/go_leaf.txt'
    )

    create_onto_protein_data(
        fin_go_graph_path='data/onto_protein_data/go_graph.txt',
        fin_go_detail_path='data/onto_protein_data/go_detail.txt',
        fin_goa_path='data/onto_protein_data/protein_go_triplet',
        fin_protein_seq_path='data/onto_protein_data/protein_seq_map.txt',
        fout_path='data/pretrain_data'
    )
