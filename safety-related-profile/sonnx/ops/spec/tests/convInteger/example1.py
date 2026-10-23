import onnx
import onnxruntime as ort
import numpy as np

def test_conv_integer_overflow():
    # Configuration
    N, C, H, W = 1, 1, 3, 3  # Batch, Channels, Height, Width
    M, kH, kW = 1, 3, 3      # Output channels, Kernel height, width
    
    # Créer des valeurs qui causent un overflow
    # Utiliser uint8 pour avoir des valeurs proches de 255
    x_data = np.full((N, C, H, W), 255, dtype=np.uint8)
    w_data = np.full((M, C, kH, kW), 255, dtype=np.uint8)
    
    # Zero points à 0 pour simplifier
    x_zero_point = np.array([0], dtype=np.uint8)
    w_zero_point = np.array([0], dtype=np.uint8)
    
    # Calcul manuel pour vérifier
    # Chaque multiplication: (255-0) * (255-0) = 65025
    # 9 multiplications (3x3): 65025 * 9 = 585225
    # Ceci est bien dans l'int32 (2,147,483,647)
    print(f"Valeur max sans dépassement: 255*255*9 = {255*255*9}")
    
    # Créer le modèle ONNX
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.UINT8, [N, C, H, W])
    W = onnx.helper.make_tensor_value_info('W', onnx.TensorProto.UINT8, [M, C, kH, kW])
    X_zero = onnx.helper.make_tensor_value_info('X_zero', onnx.TensorProto.UINT8, [1])
    W_zero = onnx.helper.make_tensor_value_info('W_zero', onnx.TensorProto.UINT8, [1])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.INT32, [N, M, 1, 1])
    
    node = onnx.helper.make_node(
        'ConvInteger',
        inputs=['X', 'W', 'X_zero', 'W_zero'],
        outputs=['Y'],
        kernel_shape=[kH, kW],
        strides=[1, 1],
        pads=[0, 0, 0, 0]
    )
    
    graph = onnx.helper.make_graph(
        [node],
        'conv_integer_test',
        [X, W, X_zero, W_zero],
        [Y]
    )
    
    model = onnx.helper.make_model(graph, producer_name='test')
    onnx.checker.check_model(model)
    
    # Exécuter avec ONNX Runtime
    sess = ort.InferenceSession(model.SerializeToString())
    outputs = sess.run(None, {
        'X': x_data,
        'W': w_data,
        'X_zero': x_zero_point,
        'W_zero': w_zero_point
    })
    
    print(f"Résultat: {outputs[0]}")
    print(f"Valeur attendue: {255*255*9}")
    print(f"Résultat dans int32? {outputs[0][0,0,0,0] < 2**31}")
    
    return outputs[0]

# Test 1: Sans overflow
result1 = test_conv_integer_overflow()

def test_conv_integer_overflow_large():
    """Test avec plus de multiplications pour causer un overflow"""
    N, C, H, W = 1, 1, 100, 100  # Grande image
    M, kH, kW = 1, 100, 100      # Grand kernel
    
    # Valeurs maximales uint8
    x_data = np.full((N, C, H, W), 255, dtype=np.uint8)
    w_data = np.full((M, C, kH, kW), 255, dtype=np.uint8)
    
    # Zero points à 0
    x_zero_point = np.array([0], dtype=np.uint8)
    w_zero_point = np.array([0], dtype=np.uint8)
    
    # Calcul manuel
    # 100*100 = 10000 multiplications de 65025
    # Résultat: 65025 * 10000 = 650,250,000 (dans int32)
    # Pas encore overflow
    print(f"Valeur max sans dépassement: 255*255*10000 = {255*255*10000}")
    print(f"Limite int32: {2**31 - 1}")
    
    # Créer le modèle
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.UINT8, [N, C, H, W])
    W = onnx.helper.make_tensor_value_info('W', onnx.TensorProto.UINT8, [M, C, kH, kW])
    X_zero = onnx.helper.make_tensor_value_info('X_zero', onnx.TensorProto.UINT8, [1])
    W_zero = onnx.helper.make_tensor_value_info('W_zero', onnx.TensorProto.UINT8, [1])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.INT32, [N, M, 1, 1])
    
    node = onnx.helper.make_node(
        'ConvInteger',
        inputs=['X', 'W', 'X_zero', 'W_zero'],
        outputs=['Y'],
        kernel_shape=[kH, kW],
        strides=[1, 1],
        pads=[0, 0, 0, 0]
    )
    
    graph = onnx.helper.make_graph(
        [node],
        'conv_integer_overflow_test',
        [X, W, X_zero, W_zero],
        [Y]
    )
    
    model = onnx.helper.make_model(graph, producer_name='test')
    onnx.checker.check_model(model)
    
    try:
        sess = ort.InferenceSession(model.SerializeToString())
        outputs = sess.run(None, {
            'X': x_data,
            'W': w_data,
            'X_zero': x_zero_point,
            'W_zero': w_zero_point
        })
        print(f"Résultat: {outputs[0][0,0,0,0]}")
        print(f"Overflow? {outputs[0][0,0,0,0] > 2**31 - 1}")
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")

test_conv_integer_overflow_large()

def test_conv_integer_custom_overflow():
    """Test avec des valeurs qui causent un overflow"""
    N, C, H, W = 1, 1, 64, 64
    M, kH, kW = 1, 64, 64
    
    # Mélange de valeurs pour maximiser le résultat
    x_data = np.random.randint(200, 256, size=(N, C, H, W), dtype=np.uint8)
    w_data = np.random.randint(200, 256, size=(M, C, kH, kW), dtype=np.uint8)
    
    x_zero_point = np.array([0], dtype=np.uint8)
    w_zero_point = np.array([0], dtype=np.uint8)
    
    # Calculer le résultat maximum possible
    max_result = 255 * 255 * 64 * 64
    print(f"Résultat maximum possible: {max_result}")
    print(f"Limite int32: {2**31 - 1}")
    print(f"Dépasse la limite? {max_result > 2**31 - 1}")
    
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.UINT8, [N, C, H, W])
    W = onnx.helper.make_tensor_value_info('W', onnx.TensorProto.UINT8, [M, C, kH, kW])
    X_zero = onnx.helper.make_tensor_value_info('X_zero', onnx.TensorProto.UINT8, [1])
    W_zero = onnx.helper.make_tensor_value_info('W_zero', onnx.TensorProto.UINT8, [1])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.INT32, [N, M, 1, 1])
    
    node = onnx.helper.make_node(
        'ConvInteger',
        inputs=['X', 'W', 'X_zero', 'W_zero'],
        outputs=['Y'],
        kernel_shape=[kH, kW],
        strides=[1, 1],
        pads=[0, 0, 0, 0]
    )
    
    graph = onnx.helper.make_graph(
        [node],
        'conv_integer_overflow_custom',
        [X, W, X_zero, W_zero],
        [Y]
    )
    
    model = onnx.helper.make_model(graph, producer_name='test')
    onnx.checker.check_model(model)
    
    try:
        sess = ort.InferenceSession(model.SerializeToString())
        outputs = sess.run(None, {
            'X': x_data,
            'W': w_data,
            'X_zero': x_zero_point,
            'W_zero': w_zero_point
        })
        
        result = outputs[0][0,0,0,0]
        print(f"Résultat: {result}")
        print(f"En hexadécimal: {hex(result)}")
        
        # Vérifier si c'est un overflow (modulo 2^32)
        expected = np.sum((x_data.astype(np.int32) - 0) * (w_data.astype(np.int32) - 0))
        mod_result = expected % (2**32)
        if mod_result > 2**31 - 1:
            mod_result = mod_result - 2**32
        
        print(f"Résultat attendu (modulo): {mod_result}")
        print(f"Résultat ONNX: {result}")
        print(f"Match avec modulo? {result == mod_result}")
        
        # Vérifier le comportement
        if result < 0 and expected > 0:
            print("⚠️ Overflow détecté - résultat négatif!")
        elif abs(result) != expected and expected < 2**31:
            print("⚠️ Comportement non standard!")
            
    except Exception as e:
        print(f"Erreur: {e}")
        # L'erreur pourrait indiquer que l'overflow n'est pas supporté

test_conv_integer_custom_overflow()

def test_overflow_scenarios():
    """Test plusieurs scénarios d'overflow"""
    scenarios = [
        # (H, W, kH, kW, use_int8)
        (32, 32, 32, 32, False),  # 1024 multiplications
        (64, 64, 64, 64, False),  # 4096 multiplications
        (128, 128, 128, 128, False),  # 16384 multiplications
        (256, 256, 256, 256, False),  # 65536 multiplications - Sûr de causer overflow
    ]
    
    for H, W, kH, kW, use_int8 in scenarios:
        print(f"\n=== Test avec {H}x{W}, kernel {kH}x{kW} ===")
        
        N, C, M = 1, 1, 1
        
        # Créer des données avec des valeurs élevées
        if use_int8:
            x_data = np.full((N, C, H, W), 127, dtype=np.int8)
            w_data = np.full((M, C, kH, kW), 127, dtype=np.int8)
            dtype = np.int8
        else:
            x_data = np.full((N, C, H, W), 255, dtype=np.uint8)
            w_data = np.full((M, C, kH, kW), 255, dtype=np.uint8)
            dtype = np.uint8
        
        x_zero_point = np.array([0], dtype=dtype)
        w_zero_point = np.array([0], dtype=dtype)
        
        # Calculer le résultat attendu
        if use_int8:
            max_val = 127
        else:
            max_val = 255
        
        total_mul = H * W
        expected = max_val * max_val * total_mul
        print(f"Multiplications: {total_mul}")
        print(f"Valeur attendue: {expected}")
        print(f"Dépasse int32? {expected > 2**31 - 1}")
        
        # Créer le modèle
        X = onnx.helper.make_tensor_value_info('X', 
            onnx.TensorProto.INT8 if use_int8 else onnx.TensorProto.UINT8, 
            [N, C, H, W])
        W = onnx.helper.make_tensor_value_info('W',
            onnx.TensorProto.INT8 if use_int8 else onnx.TensorProto.UINT8,
            [M, C, kH, kW])
        X_zero = onnx.helper.make_tensor_value_info('X_zero',
            onnx.TensorProto.INT8 if use_int8 else onnx.TensorProto.UINT8,
            [1])
        W_zero = onnx.helper.make_tensor_value_info('W_zero',
            onnx.TensorProto.INT8 if use_int8 else onnx.TensorProto.UINT8,
            [1])
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.INT32, [N, M, 1, 1])
        
        node = onnx.helper.make_node(
            'ConvInteger',
            inputs=['X', 'W', 'X_zero', 'W_zero'],
            outputs=['Y'],
            kernel_shape=[kH, kW],
            strides=[1, 1],
            pads=[0, 0, 0, 0]
        )
        
        graph = onnx.helper.make_graph(
            [node],
            f'conv_overflow_{H}_{W}',
            [X, W, X_zero, W_zero],
            [Y]
        )
        
        model = onnx.helper.make_model(graph, producer_name='test')
        
        try:
            sess = ort.InferenceSession(model.SerializeToString())
            outputs = sess.run(None, {
                'X': x_data,
                'W': w_data,
                'X_zero': x_zero_point,
                'W_zero': w_zero_point
            })
            
            result = outputs[0][0,0,0,0]
            print(f"Résultat ONNX: {result}")
            print(f"En hex: {hex(result & 0xFFFFFFFF)}")
            
            # Vérifier le modulo 2^32
            mod_expected = expected % (2**32)
            if mod_expected > 2**31 - 1:
                mod_expected = mod_expected - 2**32
            
            if expected > 2**31 - 1:
                print(f"Résultat modulo 2^32: {mod_expected}")
                if result == mod_expected:
                    print("✅ Overflow traité avec modulo 2^32")
                else:
                    print("⚠️ Comportement différent du modulo")
            else:
                if result == expected:
                    print("✅ Pas d'overflow, résultat correct")
                else:
                    print(f"❌ Erreur: attendu {expected}, obtenu {result}")
                    
        except Exception as e:
            print(f"❌ Échec: {e}")

test_overflow_scenarios()