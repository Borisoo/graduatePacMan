using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraFollow : MonoBehaviour
{

    public Transform playerTransform; // 移动的物体
    public Vector3 deviation; // 偏移量

    void Start()
    {
        deviation = transform.position - playerTransform.position; // 初始物体与相机的偏移量=相机的位置 - 移动物体的偏移量
    }

    void LateUpdate()
    {
        transform.position = playerTransform.position + deviation; // 相机的位置 = 移动物体的位置 + 偏移量

    }
}
