
using UnityEngine;
using System.Collections;
using UnityEngine.UI;

public class enemy2 : PlayObject
{
    public PlayObject m_pursuiter;  
    public PlayObject m_pursuitTarget;  //player
    public GameObject AimPredictionPos;
 
    // Use this for initialization
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        //  Vector2 moveVec = AI_Seek(m_pursuitTarget.Position);
        Vector2 moveVec = AI_PredictionPursuit();
        m_pursuiter.moveVx += moveVec.x;
        m_pursuiter.moveVz += moveVec.y;
        m_pursuiter.Move(1, true);
    }

    Vector2 AI_PredictionPursuit()
    {
        Vector2 ToPursuit = new Vector2(m_pursuitTarget.transform.position.x, m_pursuitTarget.transform.position.z) - m_pursuiter.Position;
        float RelativeHeading = DotProduct(m_pursuiter.vHeading, m_pursuitTarget.vHeading);
        // 두 점 사이의 거리 계산
        float distance = ToPursuit.magnitude;

        //Method 2: direction
        if (DotProduct(ToPursuit, m_pursuiter.vHeading) > 0 && RelativeHeading < -0.75f)
        {
            // Debug.Log("RelativeHeading:" + RelativeHeading);
            return AI_Seek(new Vector2(m_pursuitTarget.transform.position.x, m_pursuitTarget.transform.position.z));
        }

        float toPursuitLenght = Mathf.Sqrt(ToPursuit.x * ToPursuit.x + ToPursuit.y * ToPursuit.y);
        float LookAheadTime = toPursuitLenght / (m_pursuiter.MaxSpeed + 5);//速度

        Vector2 predictionPos = new Vector2(m_pursuitTarget.transform.position.x, m_pursuitTarget.transform.position.z) + m_pursuitTarget.Velocity * LookAheadTime;

        AimPredictionPos.transform.position = new Vector3(predictionPos.x, 0, predictionPos.y);
        
        return AI_Seek(predictionPos);
    }

    Vector2 AI_Seek(Vector2 TargetPos)
    {
        // Debug.Log("TargetPosx:" + TargetPos.x + "TargetPosy:" + TargetPos.y);
        Vector2 DesiredVelocity = (TargetPos - m_pursuiter.Position).normalized * m_pursuiter.MaxSpeed;
        DesiredVelocity = DesiredVelocity - m_pursuiter.Velocity;
        return DesiredVelocity;
    }

    float DotProduct(Vector2 A, Vector2 B)
    {
        return A.x * B.x + A.y * B.y;
    }
}