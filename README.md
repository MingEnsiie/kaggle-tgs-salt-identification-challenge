# Build tf-records

/usr/bin/python encode_tf_records.py \
--image-folder ../data/train/images \
--mask-folder ../data/train/masks \
--image-list ../data/train/trainset.txt \
--output-path ../data/train/tf-records/ \
--phase train


/usr/bin/python encode_tf_records.py \
--image-folder ../data/train/images \
--mask-folder ../data/train/masks \
--image-list ../data/train/evalset.txt \
--output-path ../data/train/tf-records/ \
--phase eval

# Train network

/usr/bin/python train_cnn.py \
--model_dir ./trained_models/v1/ \
--train_metadata ../data/train/tf-records/train* \
--eval_metadata ../data/train/tf-records/eval* \
--total_steps 1 \
--batch_size 2 \
--learning_rate 1e-3 \
--reg_val 1e-4 \
--batch_prefetch 2


/usr/bin/python train_cnn.py \
--model_dir ./trained_models/v1/ \
--train_metadata ../data/train/tf-records/train* \
--eval_metadata ../data/train/tf-records/eval* \
--total_steps 100000 \
--batch_size 32 \
--learning_rate 1e-4 \
--reg_val 1e-4 \
--batch_prefetch 3