def get_ratio(bit_config):
    params = {
        'word_embeddings': 23440896, 'position_embeddings': 393216,
        'layer_0_attention_query': 589824, 'layer_0_attention_key': 589824, 'layer_0_attention_value': 589824,
        'layer_0_attention_output': 589824, 'layer_0_intermediate': 2359296, 'layer_0_output': 2359296,
        'layer_1_attention_query': 589824, 'layer_1_attention_key': 589824, 'layer_1_attention_value': 589824,
        'layer_1_attention_output': 589824, 'layer_1_intermediate': 2359296, 'layer_1_output': 2359296,
        'layer_2_attention_query': 589824, 'layer_2_attention_key': 589824, 'layer_2_attention_value': 589824,
        'layer_2_attention_output': 589824, 'layer_2_intermediate': 2359296, 'layer_2_output': 2359296,
        'layer_3_attention_query': 589824, 'layer_3_attention_key': 589824, 'layer_3_attention_value': 589824,
        'layer_3_attention_output': 589824, 'layer_3_intermediate': 2359296, 'layer_3_output': 2359296,
        'layer_4_attention_query': 589824, 'layer_4_attention_key': 589824, 'layer_4_attention_value': 589824,
        'layer_4_attention_output': 589824, 'layer_4_intermediate': 2359296, 'layer_4_output': 2359296,
        'layer_5_attention_query': 589824, 'layer_5_attention_key': 589824, 'layer_5_attention_value': 589824,
        'layer_5_attention_output': 589824, 'layer_5_intermediate': 2359296, 'layer_5_output': 2359296,
        'pooler': 589824,
    }
    fenzi = 31.0 * 768.0 * 32.0 + 6.0 * 3072.0 * 32.0 + 1536.0 * 32.0 + 26.0 * 768 * 32.0
    fenmu = 31.0 * 768.0 * 32.0 + 6.0 * 3072.0 * 32.0 + 1536.0 * 32.0 + 26.0 * 768 * 32.0

    for key in params:
        fenzi += params[key] * bit_config[key]['weight_bits']
        fenmu += params[key] * 32
    ratio = fenzi / fenmu
    # print(ratio)
    return ratio


def str2dict(bit_config, bits_str):
    bits_str = bits_str.replace('epoch12 ', '').replace('epoch3 ', '')
    name_list = ['word_embeddings', 'position_embeddings',
                 'layer_0_attention_query', 'layer_0_attention_key', 'layer_0_attention_value',
                 'layer_0_attention_output', 'layer_0_intermediate', 'layer_0_output',
                 'layer_1_attention_query', 'layer_1_attention_key', 'layer_1_attention_value',
                 'layer_1_attention_output', 'layer_1_intermediate', 'layer_1_output',
                 'layer_2_attention_query', 'layer_2_attention_key', 'layer_2_attention_value',
                 'layer_2_attention_output', 'layer_2_intermediate', 'layer_2_output',
                 'layer_3_attention_query', 'layer_3_attention_key', 'layer_3_attention_value',
                 'layer_3_attention_output', 'layer_3_intermediate', 'layer_3_output',
                 'layer_4_attention_query', 'layer_4_attention_key', 'layer_4_attention_value',
                 'layer_4_attention_output', 'layer_4_intermediate', 'layer_4_output',
                 'layer_5_attention_query', 'layer_5_attention_key', 'layer_5_attention_value',
                 'layer_5_attention_output', 'layer_5_intermediate', 'layer_5_output',
                 'pooler',
                 ]
    for i, key in enumerate(name_list):
        bit_config[key]['weight_bits'] = int(bits_str[i])
    return bit_config


def set_layer_bit(bit_config, name, bit):
    for key in bit_config:
        if 'weight_bits' in bit_config[key]:
            bit_config[key]['weight_bits'] = 8
    bit_config[name]['weight_bits'] = bit
    return bit_config


bit_config = {
    'word_embeddings': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'position_embeddings': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'token_type_embeddings': {'mode': 'none', 'activation_bits': 8, 'weight_bits': 8},
    'layer_0_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_0_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_0_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_0_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_0_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_0_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_0_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_0_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_0_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_0_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_1_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_1_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_1_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_1_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_1_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_1_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_1_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_1_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_1_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_1_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_2_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_2_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_2_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_2_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_2_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_2_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_2_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_2_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_2_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_2_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_3_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_3_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_3_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_3_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_3_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_3_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_3_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_3_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_3_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_3_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_4_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_4_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_4_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_4_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_4_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_4_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_4_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_4_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_4_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_4_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_5_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_5_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_5_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_5_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_5_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_5_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_5_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_5_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_5_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_5_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_6_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_6_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_6_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_6_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_6_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_6_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_6_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_6_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_6_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_6_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_7_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_7_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_7_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_7_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_7_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_7_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_7_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_7_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_7_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_7_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_8_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_8_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_8_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_8_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_8_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_8_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_8_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_8_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_8_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_8_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_9_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_9_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_9_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_9_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_9_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_9_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_9_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_9_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_9_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_9_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_10_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_10_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_10_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_10_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_10_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_10_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_10_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_10_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_10_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_10_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_11_attention_query': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_11_attention_key': {'mode': 'ema', 'activation_bits': 8, 'weight_bits': 8},
    'layer_11_attention_value': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_11_attention_query_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_11_attention_key_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_11_attention_value_layer': {'mode': 'ema', 'activation_bits': 8},
    'layer_11_attention_probs': {'mode': 'ema', 'activation_bits': 8},
    'layer_11_attention_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_11_intermediate': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'layer_11_output': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8},
    'pooler': {'mode': 'ema', 'requantize_output': False, 'activation_bits': 8, 'weight_bits': 8}
}
for key in bit_config:
    if 'weight_bits' in bit_config[key]:
        bit_config[key]['weight_bits'] = 2
# bit_config = str2dict(bit_config, '567754777355677376753466664456744537334')


if __name__ == '__main__':
    # bit_config = str2dict(bit_config, '567754777355677376753466664456744537334')
    # for key in bit_config:
    #     if 'weight_bits' in bit_config[key]:
    #         # print(key)
    #         print(key, bit_config[key])
    ratio = get_ratio(bit_config)
    print(ratio)
