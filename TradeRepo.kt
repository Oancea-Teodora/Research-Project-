package com.example.myapp1

object TradeRepo {
    val items: MutableList<Trade> = mutableListOf()
    fun indexOf(id: String) = items.indexOfFirst { it.id == id }
}