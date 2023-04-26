import tensorflow as tf
from kgcn import layers
from kgcn.default_model import DefaultModel
import tensorflow.contrib.keras as K


def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)), axis=0)
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y, axis=0))), axis=0)
    r2 = tf.subtract(1.0, tf.div(residual, total))
    return r2


class GraphMaxGather(layers.Layer):
    def __init__(self, **kwargs):
        super(GraphMaxGather, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        super(GraphMaxGather, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.reduce_max(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class GNN(DefaultModel):
    def build_placeholders(self, info, config, batch_size, **kwargs):
        # input data types (placeholders) of this neural network
        keys = [
            "adjs",
            "labels",
            "mask",
            "enabled_node_nums",
            "is_train",
            "dropout_rate",
            "features",
        ]
        self.get_placeholders(info, config, batch_size, keys, **kwargs)
        self.placeholders["edge_feats"] = tf.placeholder(
            tf.float32,
            shape=(batch_size, info.graph_node_num, info.edge_feats_dim),
            name="edge_feats",
        )
        return self.placeholders

    def build_model(self, placeholders, info, config, batch_size, **kwargs):
        adj_channel_num = info.adj_channel_num
        in_adjs = placeholders["adjs"]
        features = placeholders["features"]
        edge_feats = placeholders["edge_feats"]
        labels = placeholders["labels"]
        mask = placeholders["mask"]
        enabled_node_nums = placeholders["enabled_node_nums"]
        if config["readout"] in ["mean", "mean_max"]:
            enabled_node_nums = (
                tf.cast(placeholders["enabled_node_nums"], tf.float32) + 1e-8
            )  # prevents Zerodivision error
        is_train = placeholders["is_train"]
        dropout_rate = placeholders["dropout_rate"]
        bin_classif_labels = labels[:, 0, None]
        cat_classif_labels = tf.cast(labels[:, 1], tf.int32)
        regr_labels = labels[:, 2:]
        regr_dim = info.regr_dim
        cat_dim = info.cat_dim
        cat_weight = config["category_weight"]
        regression_weight = config["regression_weight"]

        layer = features
        if config["concat"]:
            blocks_out = []
        for _ in range(config["num_gcn_layers"]):
            res = layer
            if edge_feats.shape[2] > 0:
                edge_feats = layers.GraphDense(config["hidden_dim"] // 2)(edge_feats)
                edge_feats = tf.nn.leaky_relu(edge_feats)
                edge_feats = layers.GraphDense(config["hidden_dim"] // 2)(edge_feats)
                edge_feats = tf.nn.leaky_relu(edge_feats)
                layer = tf.concat([layer, edge_feats], axis=2)
            if config["model"] == "GCN":
                layer = layers.GraphConv(config["hidden_dim"], adj_channel_num)(
                    layer, adj=in_adjs
                )
            elif config["model"] == "GAT":
                layer = layers.GraphDense(config["hidden_dim"])(layer)
                layer = layers.GAT(adj_channel_num)(layer, adj=in_adjs)
            elif config["model"] == "GIN":
                layer = layers.GINAggregate(adj_channel_num)(layer, adj=in_adjs)
                layer = layers.GraphDense(config["hidden_dim"])(layer)
                layer = tf.nn.leaky_relu(layer)
                layer = layers.GraphDense(config["hidden_dim"])(layer)
            if config["batchnorm"]:
                layer = layers.GraphBatchNormalization()(
                    layer,
                    max_node_num=info.graph_node_num,
                    enabled_node_nums=tf.cast(enabled_node_nums, tf.int32),
                )
            if config["residual"]:
                if layer.shape != res.shape:
                    res = layers.GraphDense(config["hidden_dim"])(res)
                layer = res + layer
            layer = tf.nn.leaky_relu(layer)
            layer = K.layers.Dropout(dropout_rate)(layer, training=is_train)
            if config["max_pool"]:
                layer = layers.GraphMaxPooling(adj_channel_num)(layer, adj=in_adjs)
            if config["concat"]:
                blocks_out.append(layer)
        layer = layers.GraphDense(config["hidden_dim"])(layer)
        layer = tf.nn.leaky_relu(layer)

        features_out = layers.GraphGather()(layer)
        if config["concat"]:
            blocks_out[-1] = layer
            read_out = [layers.GraphGather()(layer) for layer in blocks_out]
            features_out = tf.concat(read_out, axis=1)

        if config["readout"] in ["mean", "mean_max"]:
            features_out = features_out / tf.reshape(
                enabled_node_nums, [-1, 1]
            )  # Mean readout instead of sum

        if config["readout"] in ["mean_max", "sum_max"]:
            global_max_pool = GraphMaxGather()(layer)
            features_out = tf.concat([features_out, global_max_pool], axis=1)

        features_out = K.layers.Dense(config["pred_dim"])(features_out)
        features_out = tf.nn.leaky_relu(features_out)
        features_out = K.layers.Dropout(dropout_rate)(features_out, training=is_train)
        logits = K.layers.Dense(1)(features_out)
        cat_logits = K.layers.Dense(cat_dim)(features_out)
        regr_out = tf.sigmoid(K.layers.Dense(regr_dim)(features_out))

        bin_classif_pred = tf.sigmoid(logits, name="output")
        categoric_pred = tf.keras.activations.softmax(cat_logits)
        prediction = tf.concat([bin_classif_pred, categoric_pred, regr_out], axis=1)

        ###
        # computing cost and metrics
        mask_reshape = tf.reshape(mask, [-1, 1])
        cost = mask_reshape * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=bin_classif_labels, logits=logits
        )
        cat_cost = mask * tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=cat_classif_labels, logits=cat_logits
        )
        reg_cost = mask_reshape * tf.squared_difference(regr_out, regr_labels)

        cost += regression_weight * reg_cost + cat_weight * tf.reshape(
            cat_cost, [-1, 1]
        )
        cost_sum = tf.reduce_sum(cost)
        cost_opt = cost_sum / tf.reduce_sum(mask)

        correct_count = mask_reshape * tf.cast(
            tf.equal(
                tf.cast(tf.math.greater(bin_classif_pred, 0.5), tf.float32),
                bin_classif_labels,
            ),
            tf.float32,
        )
        correct_count_steps = mask * tf.cast(
            tf.equal(tf.cast(tf.argmax(cat_logits, 1), tf.int32), cat_classif_labels),
            tf.float32,
        )
        accuracy = tf.reduce_sum(correct_count) / tf.reduce_sum(mask)
        accuracy_step = tf.reduce_sum(correct_count_steps) / tf.reduce_sum(mask)

        metrics = {}
        metrics["correct_count"] = tf.reduce_sum(correct_count)
        metrics["correct_count_steps"] = tf.reduce_sum(correct_count_steps)
        metrics["accuracy"] = accuracy
        metrics["accuracy_step"] = accuracy_step
        metrics["loss"] = cost_opt
        metrics["mse"] = tf.reduce_mean(reg_cost)
        metrics["mae"] = tf.keras.losses.MeanAbsoluteError()(regr_out, regr_labels)
        metrics["r2"] = R_squared(regr_labels, regr_out)
        self.out = logits
        return self, prediction, cost_opt, cost_sum, metrics
