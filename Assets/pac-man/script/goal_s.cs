using UnityEngine;
using System.Collections;
using UnityEngine.UI;

public class goal_s : PlayObject
{
    public PlayObject goal; // goal is the goal object
    public GameObject Aim; // Aim is the player
    public float vecChangeRate; // 벡터 변화율
    public float MaxSpeedofGoal;
    public float FleeDistance = 10f; // 도망칠 거리 기준
    public bool isFleeing = false; // isFleeing 상태를 나타내는 변수
    
    Vector2 AimPos;

    // Use this for initialization
    void Start()
    {
        AimPos = new Vector2(Aim.transform.position.x, Aim.transform.position.z); //AimPos is the position of the player
    }

    // Update is called once per frame
    void Update()
    {
        AimPos = new Vector2(Aim.transform.position.x, Aim.transform.position.z);

        Vector2 moveVec = AI_UNSeek(AimPos); // moveVec is 0 if the distance is > FleeDistance(10f), 
        // otherwise it is the vector pointing away from the player
        float length = Mathf.Sqrt(moveVec.x * moveVec.x + moveVec.y * moveVec.y); 

        isFleeing = length != 0 ? true : false;
        if (length != 0) // 길이가 0이 아니면(<10f) (즉, 도망칠 필요가 있으면 )
        {
            goal.moveVx += vecChangeRate * moveVec.x / length;
            goal.moveVz += vecChangeRate * moveVec.y / length;
        }

        // 카메라 뷰 영역 밖으로 나가지 않도록 위치 보정
        KeepInsideViewport();
        goal.Move(1, true);
    }

    void KeepInsideViewport()
    {
        Vector3 viewPos = Camera.main.WorldToViewportPoint(goal.transform.position);
        viewPos.x = Mathf.Clamp(viewPos.x, 0.05f, 0.95f);
        viewPos.y = Mathf.Clamp(viewPos.y, 0.05f, 0.95f);
        goal.transform.position = Camera.main.ViewportToWorldPoint(viewPos);
    }

    Vector2 AI_UNSeek(Vector2 targetPos) // 도망치는 행동
    {
        Vector2 disPos = goal.Position - targetPos;
        float distance = Mathf.Sqrt(disPos.x * disPos.x + disPos.y * disPos.y);
        if (distance > FleeDistance) // 도망칠 거리를 벗어나면 도망치지 않음, 추척하는건가요 ㅋㅋ
        {
            return Vector2.zero; //도망치지 않으니 0 벡터 반환
        }
        // 목표 위치로부터 멀어지는 방향으로 속도 벡터 계산, 도망치는건가요 ㅋㅋ
        disPos = disPos.normalized * MaxSpeedofGoal - goal.Velocity;
        return disPos;
    }
}
