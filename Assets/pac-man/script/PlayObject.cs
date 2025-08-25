using UnityEngine;
using System.Collections;

public class PlayObject : MonoBehaviour
{
    [HideInInspector]
    public float moveVx;//x����ķ��ٶ�
    [HideInInspector]
    public float moveVz;//y����ķ��ٶ�
    public float MaxSpeed;

    /// <summary>
    /// 2ά����(x,y)
    /// </summary>
    public Vector2 Position
    {
        get
        {
            return new Vector2(this.transform.position.x, this.transform.position.z);
        }
    }
    private Vector2 _vHeading;
    /// <summary>
    /// //���õ�����ǰ������Ĺ�һ������m_vHeading
    /// </summary>
    public Vector2 vHeading
    {
        get
        {
            float length = Mathf.Sqrt(moveVx * moveVx + moveVz * moveVz);
            if (length != 0)
            {
                _vHeading.x = moveVx / length;
                _vHeading.y = moveVz / length;
            }
            return _vHeading;
        }
    }
    private Vector2 _vSide;
    /// <summary>
    /// ǰ������Ĵ�ֱ����
    /// </summary>
    public Vector2 vSide
    {
        get
        {
            _vSide.x = -vHeading.y;
            _vSide.y = vHeading.x;
            return _vSide;
        }
    }

    /// <summary>
    /// �ٶ�����
    /// </summary>
    public Vector2 Velocity
    {
        get
        {
            return new Vector2(moveVx, moveVz);
        }
    }
    /// <summary>
    /// �ٶȱ���
    /// </summary>
    public float Speed
    {
        get
        {
            return Mathf.Sqrt(moveVx * moveVx + moveVz * moveVz);
        }
    }
    public float MaxSpeedRate;
    // Use this for initialization
    void Start()
    {


    }

    // Update is called once per frame
    void Update()
    {

    }

    public void Move(float speedRate, bool isLookAtVelocityVector)
    {
        this.transform.position += new Vector3(moveVx * Time.deltaTime, 0, moveVz * Time.deltaTime) * speedRate;
        //  Debug.Log("x:" + m_postion.x + "y:" + m_postion.y);
        //���������ĳ����ǵ������ٶ�ʸ���ϳɷ���һ��
        if (isLookAtVelocityVector)
        {
            LookAtVelocityVector();
        }
    }
    /// <summary>
    /// ʹ������ʼ�ճ���ʸ���ٶȵķ���
    /// </summary>
    // void LookAtVelocityVector()
    // {
    //     float yAngles = Mathf.Atan(moveVx / moveVz) * (-180 / Mathf.PI);
    //     if (moveVz == 0)
    //     {
    //         yAngles = moveVx > 0 ? -90 : 90;
    //         //�������ļ���ǶȲ�ͬ���ǣ��������moveVx==0�Ķ����жϣ����������ڲ����Ƶ�ʱ�򱣳�ԭ״̬
    //         if (moveVx == 0)
    //         {
    //             yAngles = this.transform.rotation.eulerAngles.y;
    //         }
    //     }

    //     if (moveVz < 0)
    //     {
    //         yAngles = yAngles - 180;
    //     }
    //     Vector3 tempAngles = new Vector3(0, yAngles,0);
    //     Quaternion tempQua = this.transform.rotation;
    //     tempQua.eulerAngles = tempAngles;
    //     this.transform.rotation = tempQua;
    // }
    void LookAtVelocityVector()
    {
        Vector2 forward = new Vector2(moveVx, moveVz);
        if (forward.sqrMagnitude < 0.0001f) return;   // 속도 0이면 회전 안 함

        float angle = Mathf.Atan2(forward.x, forward.y) * Mathf.Rad2Deg;
        transform.rotation = Quaternion.Euler(0, angle, 0);
    }

}

