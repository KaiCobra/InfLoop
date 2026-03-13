"""
客戶端範例：從另一個環境/專案呼叫 Text Mask Server
不需要安裝 detectron2 或任何深度學習相關套件，只需要 requests
"""
import requests
import numpy as np
import cv2
import base64
from pathlib import Path


class TextMaskClient:
    """
    Text Mask API 客戶端
    可在任何 Python 環境中使用，只需要 requests 套件
    """
    
    def __init__(self, server_url='http://localhost:8888'):
        """
        初始化客戶端
        
        Args:
            server_url: Server 地址，格式 http://host:port
        """
        self.server_url = server_url.rstrip('/')
        self._check_connection()
    
    def _check_connection(self):
        """檢查與 server 的連線"""
        try:
            response = requests.get(f'{self.server_url}/health', timeout=5)
            if response.status_code == 200:
                print(f"✓ 成功連接到 {self.server_url}")
                print(f"  服務狀態: {response.json()}")
            else:
                print(f"✗ Server 回應異常: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"✗ 無法連接到 server: {e}")
            print(f"  請確認 server 是否在 {self.server_url} 運行")
    
    def predict_from_file(self, image_path, save_path=None):
        """
        從圖片檔案獲取 mask
        
        Args:
            image_path: 圖片檔案路徑
            save_path: mask 儲存路徑（可選）
        
        Returns:
            mask: numpy array (H, W, 3)
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"圖片不存在: {image_path}")
        
        # 上傳檔案
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f'{self.server_url}/predict',
                files=files,
                timeout=30
            )
        
        if response.status_code != 200:
            raise Exception(f"API 錯誤: {response.text}")
        
        result = response.json()
        
        if not result['success']:
            raise Exception(f"處理失敗: {result.get('error', 'Unknown error')}")
        
        # 解碼 base64 mask
        mask_base64 = result['mask_base64']
        mask_bytes = base64.b64decode(mask_base64)
        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)
        
        print(f"✓ 處理完成: {result['shape']}, {result['num_text_regions']} 個文字區域")
        
        # 儲存 mask
        if save_path:
            cv2.imwrite(str(save_path), mask)
            print(f"✓ Mask 已儲存至: {save_path}")
        
        return mask
    
    def predict_from_array(self, image_array, save_path=None):
        """
        從 numpy array 獲取 mask
        
        Args:
            image_array: numpy array (H, W, 3) BGR 格式
            save_path: mask 儲存路徑（可選）
        
        Returns:
            mask: numpy array (H, W, 3)
        """
        # 編碼圖片為 base64
        _, encoded = cv2.imencode('.png', image_array)
        image_base64 = base64.b64encode(encoded).decode('utf-8')
        
        # 發送請求
        response = requests.post(
            f'{self.server_url}/predict',
            json={'image_base64': image_base64},
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"API 錯誤: {response.text}")
        
        result = response.json()
        
        if not result['success']:
            raise Exception(f"處理失敗: {result.get('error', 'Unknown error')}")
        
        # 解碼 mask
        mask_base64 = result['mask_base64']
        mask_bytes = base64.b64decode(mask_base64)
        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)
        
        print(f"✓ 處理完成: {result['shape']}, {result['num_text_regions']} 個文字區域")
        
        if save_path:
            cv2.imwrite(str(save_path), mask)
            print(f"✓ Mask 已儲存至: {save_path}")
        
        return mask
    
    def predict_file_direct(self, image_path, save_path):
        """
        獲取 mask 並直接儲存為檔案（不經過 base64 轉換）
        適合大圖片，效能較好
        
        Args:
            image_path: 輸入圖片路徑
            save_path: 輸出 mask 路徑
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f'{self.server_url}/predict_file',
                files=files,
                timeout=30
            )
        
        if response.status_code != 200:
            raise Exception(f"API 錯誤: {response.status_code}")
        
        # 直接儲存返回的圖片
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Mask 已儲存至: {save_path}")
        
        # 讀取並返回
        mask = cv2.imread(str(save_path))
        return mask
    
    def predict_detailed(self, image_path):
        """
        獲取詳細的檢測資訊
        
        Returns:
            dict: 包含 mask, texts, scores, languages, boxes
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f'{self.server_url}/predict_detailed',
                files=files,
                timeout=30
            )
        
        if response.status_code != 200:
            raise Exception(f"API 錯誤: {response.text}")
        
        result = response.json()
        
        if not result['success']:
            raise Exception(f"處理失敗: {result.get('error', 'Unknown error')}")
        
        # 解碼 mask
        mask_base64 = result['mask_base64']
        mask_bytes = base64.b64decode(mask_base64)
        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)
        
        return {
            'mask': mask,
            'num_regions': result['num_regions'],
            'texts': result['texts'],
            'scores': result['scores'],
            'languages': result['languages'],
            'boxes': result['boxes']
        }


# ============ 使用範例 ============

def example1_simple_usage():
    """範例 1: 基本使用"""
    print("\n=== 範例 1: 基本使用 ===")
    
    # 初始化客戶端
    client = TextMaskClient('http://localhost:5000')
    
    # 處理圖片
    mask = client.predict_from_file('generated.jpg', 'output_mask.jpg')
    print(f"Mask shape: {mask.shape}")


def example2_pipeline_integration():
    """範例 2: Pipeline 整合"""
    print("\n=== 範例 2: Pipeline 整合 ===")
    
    client = TextMaskClient('http://localhost:5000')
    
    # 模擬生成模型產生圖片
    def your_generative_model():
        """你的生成模型"""
        # 這裡只是範例，實際上會是你的模型
        image = cv2.imread('generated.jpg')
        return image
    
    # Pipeline 流程
    for i in range(3):
        print(f"\n處理圖片 #{i+1}...")
        
        # 1. 生成圖片
        generated_image = your_generative_model()
        
        # 2. 獲取文字 mask
        mask = client.predict_from_array(generated_image)
        
        # 3. 使用 mask 做後續處理
        print(f"  - 文字區域像素: {np.sum(mask > 0)}")
        
        # 4. 儲存結果
        cv2.imwrite(f'pipeline_mask_{i+1}.jpg', mask)


def example3_detailed_info():
    """範例 3: 獲取詳細資訊"""
    print("\n=== 範例 3: 獲取詳細資訊 ===")
    
    client = TextMaskClient('http://localhost:5000')
    
    result = client.predict_detailed('generated.jpg')
    
    print(f"檢測到 {result['num_regions']} 個文字區域:")
    for i, (text, score) in enumerate(zip(result['texts'], result['scores'])):
        print(f"  區域 {i+1}: {text} (信心度: {score:.3f})")


def example4_batch_processing():
    """範例 4: 批次處理多張圖片"""
    print("\n=== 範例 4: 批次處理 ===")
    
    client = TextMaskClient('http://localhost:5000')
    
    # 模擬多張圖片
    image_files = ['generated.jpg'] * 3  # 替換成實際的圖片列表
    
    for idx, image_path in enumerate(image_files):
        if not Path(image_path).exists():
            continue
            
        print(f"\n處理 {image_path}...")
        mask = client.predict_file_direct(
            image_path, 
            f'batch_output_{idx}.jpg'
        )


def example5_error_handling():
    """範例 5: 錯誤處理"""
    print("\n=== 範例 5: 錯誤處理 ===")
    
    try:
        client = TextMaskClient('http://localhost:5000')
        mask = client.predict_from_file('generated.jpg')
        print("處理成功！")
        
    except FileNotFoundError as e:
        print(f"檔案錯誤: {e}")
    except requests.exceptions.ConnectionError:
        print("無法連接到 server，請確認 server 是否運行")
    except Exception as e:
        print(f"發生錯誤: {e}")


if __name__ == '__main__':
    print("Text Mask Client 使用範例")
    print("=" * 60)
    print("請確認 server 已啟動: python text_mask_server.py")
    print("=" * 60)
    
    try:
        # USAGE:
        example1_simple_usage()
        # example2_pipeline_integration()
        # example3_detailed_info()
        # example4_batch_processing()
        # example5_error_handling()
        
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
