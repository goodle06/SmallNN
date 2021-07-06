#pragma once

#include <map>
#include <string>

namespace DesignPatterns {


	template <typename AbstractProduct, typename KeyType, typename ProductCreator> 
	class Factory {
	public:
		bool Register(const KeyType id, ProductCreator creator);
		bool Unregister(const KeyType id);
		AbstractProduct* SpawnObject(const KeyType id);
	private:
		typedef std::map<KeyType, ProductCreator> factory_map;
		factory_map m_lib;
	};

	template<typename AbstractProduct, typename KeyType, typename ProductCreator>
	inline bool Factory<AbstractProduct, KeyType, ProductCreator>::Register(const KeyType id, ProductCreator creator)
	{
		return m_lib.insert(factory_map::value_type(id, creator)).second;
	}

	template<typename AbstractProduct, typename KeyType, typename ProductCreator>
	bool DesignPatterns::Factory<AbstractProduct, KeyType, ProductCreator>::Unregister(const KeyType id)
	{
		return m_lib.erase(id) == 1;
	}

	template<typename AbstractProduct, typename KeyType, typename ProductCreator>
	AbstractProduct* DesignPatterns::Factory<AbstractProduct, KeyType, ProductCreator>::SpawnObject(const KeyType id)
	{
		typename factory_map::const_iterator it = m_lib.find(id);
		if (it != m_lib.end())
			return it->second();
		return nullptr;
	}
	



}