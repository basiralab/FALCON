python main_Cora.py --model_type QSIGN --centrality NO --num_epoch 400
python main_Cora.py --model_type QSIGN --centrality DC --num_epoch 400
python main_Cora.py --model_type QSIGN --centrality BC --num_epoch 400
python main_Cora.py --model_type QSIGN --centrality PR --num_epoch 400
python main_Cora.py --model_type QSIGN --centrality CC --num_epoch 400
python main_Cora.py --model_type QSIGN --centrality EC --num_epoch 400
python main_Cora_coarsen.py --num_epoch 400

python main_Cora.py --model_type GCN --centrality NO --num_epoch 400
python main_Cora.py --model_type GCN --centrality DC --num_epoch 400
python main_Cora.py --model_type GCN --centrality BC --num_epoch 400
python main_Cora.py --model_type GCN --centrality PR --num_epoch 400
python main_Cora.py --model_type GCN --centrality CC --num_epoch 400
python main_Cora.py --model_type GCN --centrality EC --num_epoch 400
python main_Cora_coarsen.py --model_type GCN --num_epoch 400

python main_PubMed.py --model_type QSIGN --centrality NO --num_epoch 400
python main_PubMed.py --model_type QSIGN --centrality DC --num_epoch 400
python main_PubMed.py --model_type QSIGN --centrality BC --num_epoch 400
python main_PubMed.py --model_type QSIGN --centrality PR --num_epoch 400
python main_PubMed.py --model_type QSIGN --centrality CC --num_epoch 400
python main_PubMed.py --model_type QSIGN --centrality EC --num_epoch 400
python main_PubMed_coarsen.py --num_epoch 400

python main_PubMed.py --model_type GCN --centrality NO --num_epoch 400
python main_PubMed.py --model_type GCN --centrality DC --num_epoch 400
python main_PubMed.py --model_type GCN --centrality BC --num_epoch 400
python main_PubMed.py --model_type GCN --centrality PR --num_epoch 400
python main_PubMed.py --model_type GCN --centrality CC --num_epoch 400
python main_PubMed.py --model_type GCN --centrality EC --num_epoch 400
python main_PubMed_coarsen.py --model_type GCN --num_epoch 400
