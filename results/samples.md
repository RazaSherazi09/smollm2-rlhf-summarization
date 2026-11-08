# Task 6 Qualitative Comparisons: Base vs. SFT vs. PPO vs. DPO

### Sample 1
**Prompt (Tail):** ...e ... also I'm big on comuncating my feelings , but I'm petrified telling him his words hurt me, will result in another fight like yesterday's one. What should I do ? Because at this point I really just want to be done with him and move on ...
TL;DR:

**Human Reference:** Boyfriend said some would stuff to me and I am so deeply hurt by them, but I'm scared bringing it up will result in another argument in which he ruins me once more. *(Reward: +1.812)*

- **Base Model:** I'm not going to tell him I'm sorry, I'm not going to tell him I'm sorry, I'm not going to tell him I'm sorry, I'm not going to tell him I'm sorry, I'm not going *(Reward: -2.812)*
- **SFT (Stage 1):** boyfriend said hurtful things to me, I'm petrified to tell him I'm hurt by it, I want to hurt him and move on, but I'm afraid to. *(Reward: +0.266)*
- **PPO (Stage 3):** I want to hurt my feelings with my boyfriend and want to talk to him about what he said about me and what he did to hurt me and what I want to do with that again after he hurt me again and what I want to *(Reward: -0.471)*
- **DPO (Stage 4):** boyfriend said incredibly hurtful things to me and I'm deeply hurt by what he said, will result in another fight like yesterday's one. What should I do ? Because at this point I really just want to be done with him and move *(Reward: +4.031)*

---

### Sample 2
**Prompt (Tail):** ...ng I do, so I was just wondering if it would be ok to just straight up ask her to be my girlfriend. I've picked out somewhere private and something meaningful to say. Should I do it? Is this the best approach to the situation I'm dealing with.
TL;DR:

**Human Reference:** If we both know we like each other, and have really hit it off, can I just straight up ask her to be my girlfriend? *(Reward: +0.484)*

- **Base Model:** I'm not sure if I should ask her to be my girlfriend or not.

I'm not sure if I should ask her to be my girlfriend.

I'm not sure if I should ask her to be my girlfriend. *(Reward: -1.398)*
- **SFT (Stage 1):** I've been assured that a girl has the same feeling I have, is it alright to just ask her to be my girlfriend? *(Reward: +1.875)*
- **PPO (Stage 3):** I want to ask her to be my girlfriend but she's so scared about what she's doing and what she's doing around me and what she's saying about things she's doing that I want to do something private and better deal like she *(Reward: +0.017)*
- **DPO (Stage 4):** When you've been assured that a girl has the same feeling you have, is it alright to just ask here to be "more than friends"? I've hung out with her after school a lot and got to really know each other. *(Reward: +3.312)*

---

### Sample 3
**Prompt (Tail):** ...s really hard to realize for me that she´s gone.  
I tried to talk to her but she just said "just let it be"..  
  
Should i keep trying to talk her?Should i just wait?Maybe she will talk to me after some time?Can it be that she has a new one?
TL;DR:

**Human Reference:** Girlfriend broke up and told me "its her fault,i cant love,im not able to bond" *(Reward: +0.088)*

- **Base Model:** I dont know if she is going to be able to love me anymore.

I am not sure if she is going to be able to love me anymore.

I am not sure if she is going to be able to love *(Reward: -0.453)*
- **SFT (Stage 1):** ex broke up with me because she cant love. I dont know if she is able to love or not. *(Reward: +0.930)*
- **PPO (Stage 3):** I want to talk to my ex girlfriend who broke off with me and she said she can't love and she wants to talk to me all the time but she says she's just lost or broken off with something and she wants to talk *(Reward: +2.516)*
- **DPO (Stage 4):** ex girlfriend broke up with me because she cant love and she is "unable to bond"..is there really something like "not able to love"? Is there really something like "unable to bond" in her? Should i *(Reward: +3.484)*

---

### Sample 4
**Prompt (Tail):** ...n or Charles Schwab? I've heard good things on this sub for both options.

Main Questions:  
1. Can I keep my savings account and credit card with BoA and just close my checking?  
2. What are the pros/cons of credit unions and Charles Schwab?
TL;DR:

**Human Reference:** I want to close my BoA checking account, but keep my savings and credit card with them. Is this possible? And, which is better: a credit union or Charles Schwab? *(Reward: +2.766)*

- **Base Model:** I'm not sure what to do. I'm not sure if I should keep my savings account or credit card. I'm not sure if I should keep my savings account or credit card. I'm not sure if I should keep my savings *(Reward: +0.268)*
- **SFT (Stage 1):** I'm banking with BoA and want to close my checking account with them. What are the pros/cons of credit unions and Charles Schwab? *(Reward: +2.359)*
- **PPO (Stage 3):** I want to keep my savings account and credit card with BoA and just move to a credit union or Chase and then close my checking account with them. What are the pros/cons of that sort of thing? I want to move back *(Reward: +3.000)*
- **DPO (Stage 4):** Baccarand Checking Account Conversion to Core Checking Account with BoA and possibly moving to a credit union or Charles Schwab? Pros/Cons of BoA and Charles Schwab? *(Reward: +2.719)*

---

### Sample 5
**Prompt (Tail):** ...mean she will be back? When am I suppoed to talk to her again. This heartache and depression is eating me from the inside, and since I have been cheated on in the past, I am only assuming the worst and that somebody else could be on the scene?
TL;DR:

**Human Reference:** GF broke up with me, Don't know if she will be back or not? We have a holiday booked, should I be expecting her to come? Or should I just not go atal/go on my own? *(Reward: +1.211)*

- **Base Model:** I am not sure if I should hold out or not. I am not sure if I should be able to talk to her again.

I am not sure if I should be able to talk to her again. I am not sure *(Reward: -1.477)*
- **SFT (Stage 1):** GF broke up with me, we have a holiday booked, should I hold out? *(Reward: +1.219)*
- **PPO (Stage 3):** I want to talk to my girl of love and she is going somewhere but I can't remember where she is or what she is doing. I want to talk to her but she is going somewhere but I can't remember what she is missing *(Reward: +0.508)*
- **DPO (Stage 4):** my GF broke up with me recently and we have a holiday booked, should I hold out? Should I be worried about her coming back? She is having some rough patches and we have a holiday booked in September. Should I *(Reward: +2.938)*

---

