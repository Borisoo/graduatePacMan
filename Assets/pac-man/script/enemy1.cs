using UnityEngine;
using System.Collections;
using UnityEngine.UI;

public class enemy1 : MonoBehaviour
{
    public GameObject target;
    public float MIN_trackingRate; //追踪加速度
    public float MIN_TrackingDis;
    public float MAX_trackingVel;
    public float moveVx; //x속도
    public float moveVz;//z속도
                        // Use this for initialization
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        Debug.Log((Mathf.Atan(moveVx / moveVz) * (-180 / Mathf.PI)));

        //  LookAtTarget();
        //  this.transform.position += new Vector3(moveVx * Time.deltaTime, moveVy * Time.deltaTime, 0);
        
        Track_AIAdvanced();
    }

    /// <summary>
    /// 
    /// </summary>
    void Track_AIAdvanced()
    {
        float vx = target.transform.position.x - this.transform.position.x;
        float vz = target.transform.position.z - this.transform.position.z;

        float length = PointDistance_2D(vx, vz);
        if (length < MIN_TrackingDis)
        {
            vx = MIN_trackingRate * vx / length;
            vz = MIN_trackingRate * vz / length;
            moveVx += vx;
            moveVz += vz;

            if (Random.Range(1, 10) == 1)
            {
                vx = Random.Range(-1, 1);
                vz = Random.Range(-1, 1);
                moveVx += vx;
                moveVz += vz;
            }
            length = PointDistance_2D(moveVx, moveVz);

            if (length > MAX_trackingVel)
            {
                moveVx *= 0.75f;
                moveVz *= 0.75f;
                // moveVx *= 0.95f;
                // moveVz *= 0.95f;
            }

        }
        //If not in tracking range, random movement
        else
        {
            //원본은 enemy2가 랜덤으로 움직이게 되어있음, 일정거리 이상이 되면 미아상태로 처밖힘
            // if (Random.Range(1, 10) == 1)
            // {
            //     vx = Random.Range(-2, 2);
            //     vz = Random.Range(-2, 2);
            //     moveVx += vx;
            //     moveVz += vz;
            // }
            // length = PointDistance_2D(moveVx, moveVz);

            // if (length > MAX_trackingVel)
            // {
            //     moveVx *= 0.75f;
            //     moveVz *= 0.75f;
            // }

            // Wandering behavior
            Vector2 wander = Random.insideUnitCircle * 0.8f;   // 랜덤한 방향 벡터 생성, 크기는 0.8f
            moveVx += wander.x * Time.deltaTime;
            moveVz += wander.y * Time.deltaTime;

            // 속도 제한, 속도가 1.5f를 넘으면 정규화하여 1.5f로 맞춤
            Vector2 vel = new Vector2(moveVx, moveVz);
            if (vel.sqrMagnitude > 1.5f * 1.5f)
                vel = vel.normalized * 1.5f;

            moveVx = vel.x;
            moveVz = vel.y;
        }

        this.transform.position += new Vector3(moveVx * Time.deltaTime, 0, moveVz * Time.deltaTime);
        KeepInsideViewport();
    }

    void KeepInsideViewport()
    {
        Vector3 viewPos = Camera.main.WorldToViewportPoint(this.transform.position);
        viewPos.x = Mathf.Clamp(viewPos.x, 0.05f, 0.95f);
        viewPos.y = Mathf.Clamp(viewPos.y, 0.05f, 0.95f);
        this.transform.position = Camera.main.ViewportToWorldPoint(viewPos);
    }

    float PointDistance_2D(float x, float y)
    {
        /*x = Mathf.Abs(x);
        y = Mathf.Abs(y);
        float mn = Mathf.Min(x, y);//��ȡx,y����С����
        float result = x + y - (mn / 2) - (mn / 4) + (mn / 8);*/

        float result = Mathf.Sqrt(x * x + y * y);
        return result;
    }
}